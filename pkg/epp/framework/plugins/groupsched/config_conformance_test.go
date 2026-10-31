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

// THE EMITTER CONTRACT (exp-ops ↔ plugin config). The exp-ops renderer writes this
// plugin's `parameters:` block as YAML and the EPP decodes it with unknown fields
// REJECTED, so an emitted key that no longer exists in Config is not a warning — it is
// a CrashLoop on every arm that carries it (the residueRelease lesson, 2026-10-31).
//
// The list below is the EMITTER's side of the contract, copied from the printf blocks
// in exp-ops/lib/common.sh. It is checked against Config's STRUCT TAGS by reflection,
// so a field rename or deletion fails HERE instead of in a deploy — and it is checked
// in both directions: a Config field the emitter cannot address is also a bug (the
// knob would silently do nothing on every arm).
//
// The second half pins the ENUM VALUES: validate() rejects an unknown value and the
// EPP would CrashLoop on that arm, so a value the emitter can write but the plugin
// rejects is caught here too.
//
// CORE SHAPE (what a core arm renders):
//
// (The emitter change that renders exactly these blocks ships WITH the core; the
// campaign tree was in use when the core was cut, so the text is handed over rather
// than edited underneath a running wave.)
//
//	parameters:                     # CORE_BASE_BLOCK
//	  groupBudgetFraction: 0.95
//	  quantum: 4
//	  idleSeconds: 20
//	  ttlSeconds: 30
//	  coldEntryCapTokens: 131072
//	  inFlightCap: 64
//	  blockSizeTokens: 64
//	  capacityLivenessSeconds: 120
//	  sizing: "index"               # CORE_MODULE_BLOCK (one selector per module)
//	  entryOrder: "newcomer-first"
//	  engineChoice: "most-free-budget"
//	  rotation: "quantum"
//	  victim: "valve"
//	  safety: "none"
//	flowControl:
//	  saturationDetector:
//	    pluginRef: group           # the SAME plugin instance answers the pool gate

import (
	"reflect"
	"strings"
	"testing"
)

// emitterKeys are the group-scheduler keys exp-ops renders, by block.
var emitterKeys = map[string][]string{
	// templates/plugins.yaml.tmpl — the base parameter block (always emitted)
	"CORE_BASE_BLOCK": {"groupBudgetFraction", "quantum", "idleSeconds", "ttlSeconds",
		"coldEntryCapTokens", "inFlightCap", "blockSizeTokens", "capacityLivenessSeconds",
		"inflightReapSeconds"},
	// CORE_MODULE_BLOCK — one value per module, selected by the arm
	"CORE_MODULE_BLOCK": {"sizing", "entryOrder", "engineChoice", "rotation", "victim", "safety"},
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
		{"Safety", []string{"none", "budget-fraction"}},
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
	for block, keys := range emitterKeys {
		for _, k := range keys {
			emitted[k] = true
			if !tags[k] {
				t.Errorf("exp-ops emits %q (%s) but Config has no such field: the EPP rejects unknown fields, so every arm carrying that block CrashLoops at startup.", k, block)
			}
		}
	}
	for k := range tags {
		if !emitted[k] {
			t.Errorf("Config field %q is not emitted by exp-ops: an arm cannot set it (add it to the base/module block, or delete the field)", k)
		}
	}
	_ = emitted
}
