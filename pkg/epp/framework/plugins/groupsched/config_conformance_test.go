/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package groupsched

// THE EMITTER CONTRACT (exp-ops ↔ plugin config).

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"sigs.k8s.io/yaml"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
)

// emittedBlock is one `parameters:` block exp-ops renders: its TEXT (the printf body from exp-
// ops/lib/common.sh, at the values the shipped arms render) and the plugin whose Parameters
// struct must decode it.
type emittedBlock struct {
	block string // the exp-ops block name (common.sh / docs/gsch-core/README.md §2)
	owner string // the plugin type whose parameters these are
	proto any    // a zero Parameters value: its TYPE is the contract
	body  string // rendered `key: value` lines, exactly as the emitter prints them
}

// emittedBlocks mirrors the printf blocks in exp-ops/lib/common.sh.
var emittedBlocks = []emittedBlock{
	{
		block: "CORE_BASE_BLOCK",
		owner: PluginType,
		proto: Config{},
		body: `groupBudgetFraction: 0.95
quantum: 4
idleSeconds: 20
inFlightCap: 64
blockSizeTokens: 64
capacityLivenessSeconds: 120
inflightReapSeconds: 3700`,
	},
	{
		block: "CORE_MODULE_BLOCK",
		owner: PluginType,
		proto: Config{},
		body: `sizing: "index"
entryOrder: "newcomer-first"
engineChoice: "most-free-budget"
rotation: "quantum"
victim: "valve"
safety: "none"
executionBudget: "none"`,
	},
	{
		// The branch plugin every core arm carries (BRANCH_PLUGIN=1).
		block: "BRANCH_IDENTITY_BLOCK",
		owner: branchidentity.PluginType,
		proto: branchidentity.Parameters{},
		body: `blockSizeTokens: 64
forkOnBusy: true
stickyBusyGuard: true
retryRule: false
logRequests: true
logHeaders: false
sessionHeaders: []`,
	},
}

// blockKeys parses the emitted keys out of a rendered block, so the key check and the decode
// check can never drift apart (one list, not two).
func blockKeys(t *testing.T, b emittedBlock) []string {
	t.Helper()
	var keys []string
	for _, line := range strings.Split(b.body, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, _, ok := strings.Cut(line, ":")
		if !ok {
			t.Fatalf("%s: rendered line %q is not `key: value`", b.block, line)
		}
		keys = append(keys, strings.TrimSpace(k))
	}
	if len(keys) == 0 {
		t.Fatalf("%s: no keys parsed — the guard would pass vacuously", b.block)
	}
	return keys
}

// strictDecodeEmitted feeds a rendered parameter block through the loader's own path: YAML →
// JSON (sigs.k8s.io/yaml, as the config loader does) → strict JSON decode with unknown fields
// rejected (configloader.go hands each factory fwkplugin.StrictDecoder).
func strictDecodeEmitted(t *testing.T, what, body string, proto any) error {
	t.Helper()
	raw, err := yaml.YAMLToJSON([]byte(body))
	if err != nil {
		t.Fatalf("%s: not valid YAML (%v): %q", what, err, body)
	}
	dec := fwkplugin.StrictDecoder(raw)
	if dec == nil {
		t.Fatalf("%s: empty parameters block", what)
	}
	return dec.Decode(reflect.New(reflect.TypeOf(proto)).Interface())
}

// TestConfigEmitterConformance fails when the emitter and Config disagree — in either
// direction that can crash a deploy or silently disable a knob.
func TestConfigEmitterConformance(t *testing.T) {
	tags := map[string]bool{}
	rt := reflect.TypeOf(Config{})
	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		name := strings.Split(f.Tag.Get("json"), ",")[0]
		if name == "" || name == "-" {
			t.Fatalf("Config field %s has no json tag — the emitter cannot address it and the loader cannot set it", f.Name)
		}
		tags[name] = true
	}
	emitted := map[string]bool{}
	for _, b := range emittedBlocks {
		if b.owner != PluginType {
			continue // the exhaustive both-ways check is the core's own blocks
		}
		for _, k := range blockKeys(t, b) {
			emitted[k] = true
			if !tags[k] {
				t.Errorf("exp-ops emits %q (%s) but Config has no such field: the EPP rejects unknown fields, so every arm carrying that block CrashLoops at startup.", k, b.block)
			}
		}
	}
	for k := range tags {
		if !emitted[k] {
			t.Errorf("Config field %q is not emitted by exp-ops: an arm cannot set it (add it to the base/module block, or delete the field)", k)
		}
	}
}

// TestEmittedBlocksDecodeIntoTheirParameters is the TYPE half of the contract: every block
// exp-ops renders must decode into its plugin's Parameters struct through the loader's strict
// path.
func TestEmittedBlocksDecodeIntoTheirParameters(t *testing.T) {
	for _, b := range emittedBlocks {
		if err := strictDecodeEmitted(t, b.block, b.body, b.proto); err != nil {
			t.Errorf("exp-ops renders %s for %s but a %s parameters struct rejects it: %v\n"+
				"The EPP decodes plugin parameters strictly and fails CLOSED, so this is a CrashLoop on every arm carrying the block — the VALUE TYPE (not just the key) has to match the Go field.",
				b.block, b.owner, b.owner, err)
		}
	}
}

// TestEmittedBoolMustBeAYAMLBoolNotANumber is the residueRelease/logRequests lesson made
// structural: a numeric literal where the Go field is a bool must FAIL the guard.
// `logRequests: 1` is what `BRANCH_LOG_REQUESTS=1` rendered in the shipped docs/gsch-
// core/*.env files — and a strict decode cannot turn a YAML number into a Go bool, so a type
// mismatch has to be a build test, not a crash-looping pod.
func TestEmittedBoolMustBeAYAMLBoolNotANumber(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr bool
	}{
		{"logRequests: 1 (the shipped-env bug)", "logRequests: 1", true},
		{"logRequests: true (the fix)", "logRequests: true", false},
		{`logRequests: "true" (a quoted string is not a bool either)`, `logRequests: "true"`, true},
		{"logRequests: false", "logRequests: false", false},
		{"forkOnBusy: 1 (*bool is checked the same way)", "forkOnBusy: 1", true},
		{"forkOnBusy: true", "forkOnBusy: true", false},
		// the decoder is not rejecting everything: a field OF THAT TYPE still takes a value
		{"blockSizeTokens: 64 (int)", "blockSizeTokens: 64", false},
		{"blockSizeTokens: not-a-number (int)", "blockSizeTokens: not-a-number", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := strictDecodeEmitted(t, tc.name, tc.body, branchidentity.Parameters{})
			switch {
			case tc.wantErr && err == nil:
				t.Errorf("%q decodes cleanly but must be REJECTED: a value whose YAML type does not match the Go field is a CrashLoop at startup", tc.body)
			case !tc.wantErr && err != nil:
				t.Errorf("%q is a valid rendering but the strict decode rejects it: %v", tc.body, err)
			}
		})
	}
}

// boolEnvKnobs are the env knobs exp-ops renders into a Go BOOL field (branch-identity's
// logRequests/logHeaders and the three *bool rules).
var boolEnvKnobs = []string{
	"BRANCH_FORK_ON_BUSY",
	"BRANCH_STICKY_BUSY_GUARD",
	"BRANCH_RETRY_RULE",
	"BRANCH_LOG_REQUESTS",
	"BRANCH_LOG_HEADERS",
}

// TestShippedCoreEnvFilesUseYAMLBools pins the env files this build SHIPS to exp-ops
// (docs/gsch-core/*.env): every knob that lands in a Go bool field must be spelled
// `true`/`false`, never `1`/`0`.
func TestShippedCoreEnvFilesUseYAMLBools(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("..", "..", "..", "..", "..", "docs", "gsch-core", "*.env"))
	if err != nil {
		t.Fatalf("glob shipped core env files: %v", err)
	}
	if len(files) == 0 {
		t.Skip("docs/gsch-core/*.env is not in this checkout")
	}
	isBoolKnob := map[string]bool{}
	for _, k := range boolEnvKnobs {
		isBoolKnob[k] = true
	}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for i, line := range strings.Split(string(raw), "\n") {
			kv := strings.TrimSpace(strings.SplitN(line, "#", 2)[0])
			key, value, ok := strings.Cut(kv, "=")
			key, value = strings.TrimSpace(key), strings.TrimSpace(value)
			if !ok || !isBoolKnob[key] {
				continue
			}
			if value != "true" && value != "false" {
				t.Errorf("%s:%d: %s=%q — the emitter writes this value VERBATIM into a Go bool field "+
					"(strict decode, unknown fields and wrong types both rejected): use true|false, or every arm carrying it CrashLoops",
					f, i+1, key, value)
			}
		}
	}
}

// moduleEnvKnobs maps exp-ops' selector env knobs to the Config field each one fills.
var moduleEnvKnobs = map[string]string{
	"CORE_SIZING":           "Sizing",
	"CORE_ENTRY_ORDER":      "EntryOrder",
	"CORE_ENGINE_CHOICE":    "EngineChoice",
	"CORE_ROTATION":         "Rotation",
	"CORE_VICTIM":           "Victim",
	"CORE_SAFETY":           "Safety",
	"CORE_EXECUTION_BUDGET": "ExecutionBudget",
}

// TestShippedCoreEnvFilesUseKnownModuleValues pins the selector VALUES in the env files this
// build ships: every one must be a value buildModules accepts, because the alternative is an
// arm that refuses to start.
func TestShippedCoreEnvFilesUseKnownModuleValues(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("..", "..", "..", "..", "..", "docs", "gsch-core", "*.env"))
	if err != nil {
		t.Fatalf("glob shipped core env files: %v", err)
	}
	if len(files) == 0 {
		t.Skip("docs/gsch-core/*.env is not in this checkout")
	}
	checked := 0
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for i, line := range strings.Split(string(raw), "\n") {
			kv := strings.TrimSpace(strings.SplitN(line, "#", 2)[0])
			key, value, ok := strings.Cut(kv, "=")
			key, value = strings.TrimSpace(key), strings.TrimSpace(value)
			knob, isModule := moduleEnvKnobs[key]
			if !ok || !isModule {
				continue
			}
			cfg := defaultConfig()
			reflect.ValueOf(&cfg).Elem().FieldByName(knob).SetString(value)
			checked++
			if _, err := buildModules(cfg); err != nil {
				t.Errorf("%s:%d: %s=%q — buildModules fails closed on it, so this arm CrashLoops at startup: %v", f, i+1, key, value, err)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no module selector was checked — the guard would pass vacuously")
	}
}

// TestReapBoundsSitAboveTheHarnessTimeout pins the one number pair that has to move together:
// the request-reap bound (both the scheduler's record reap and branch identity's marker TTL)
// is the BELT over an unobserved termination, so it must sit above the CLIENT's own timeout —
// the campaign harness runs 9000 s — or the belt cuts a request that is still legitimately
// running, and the branch it belonged to takes the next request as a different chain.
func TestReapBoundsSitAboveTheHarnessTimeout(t *testing.T) {
	const harnessTimeoutSeconds = 9000
	if got := defaultConfig().reapSeconds(); got <= harnessTimeoutSeconds {
		t.Fatalf("the scheduler's reap default is %vs, at or below the %ds harness timeout: it would cut live requests", got, harnessTimeoutSeconds)
	}
}

// TestModuleValuesMatchTheEmitter pins the VALUES exp-ops can render for every module
// selector: buildModules must accept each one and reject anything else.
func TestModuleValuesMatchTheEmitter(t *testing.T) {
	emitted := []struct {
		knob   string
		values []string
	}{
		{"Sizing", []string{"index"}},
		{"EntryOrder", []string{"newcomer-first"}},
		{"EngineChoice", []string{"most-free-budget"}},
		{"Rotation", []string{"quantum"}},
		{"Victim", []string{"valve"}},
		{"Safety", []string{"none"}},
		{"ExecutionBudget", []string{"none"}},
	}
	for _, e := range emitted {
		for _, v := range e.values {
			cfg := defaultConfig()
			reflect.ValueOf(&cfg).Elem().FieldByName(e.knob).SetString(v)
			if _, err := buildModules(cfg); err != nil {
				t.Errorf("exp-ops can render %s=%q but the plugin rejects it: %v", e.knob, v, err)
			}
		}
		cfg := defaultConfig()
		reflect.ValueOf(&cfg).Elem().FieldByName(e.knob).SetString("not-a-module")
		if _, err := buildModules(cfg); err == nil {
			t.Errorf("%s: an unknown value must fail closed (a typo would otherwise measure the default twice)", e.knob)
		}
	}
}
