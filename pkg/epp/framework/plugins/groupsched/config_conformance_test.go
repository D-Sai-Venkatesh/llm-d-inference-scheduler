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

// THE EMITTER CONTRACT (exp-ops ↔ plugin config). The exp-ops renderer
// (experiments/exp-ops/lib/common.sh) writes this plugin's `parameters:` block as YAML,
// and the EPP decodes it with unknown fields REJECTED — so an emitted key that no longer
// exists in Config is not a warning, it is a CrashLoop on every arm that carries it.
//
// That is exactly what happened on 2026-10-31: the F6′ `residueRelease` knob was deleted
// from the build (F6″ is a residue-aware VICTIM ORDER, not a seat release), the emitter
// kept emitting it, and all seven B1b arms died at startup with
// `json: unknown field "residueRelease"`. Nothing on either side noticed at build time.
//
// The list below is the EMITTER's side of the contract, copied from the printf blocks in
// exp-ops/lib/common.sh (one group per block, so a failure names the block to look at).
// It is checked against Config's STRUCT TAGS, which are read by reflection — a field
// RENAME or DELETION on the build side therefore fails here instead of in a deploy, and
// the fix is either to update the emitter or (when the knob is deliberately gone) to
// delete it from both.

import (
	"reflect"
	"strings"
	"testing"
)

// emitterKeys are the group-scheduler keys exp-ops can render, by block.
var emitterKeys = map[string][]string{
	// templates/plugins.yaml.tmpl — V2_KNOBS_BLOCK (EMIT_V2_KNOBS=1)
	"V2_KNOBS_BLOCK": {"rotationMode", "quantumBusyPeriods", "valveMode", "victimOrder",
		"windowMode", "gateMode", "priorityHeader"},
	// WAVE1_KNOBS_BLOCK (EMIT_W1_KNOBS=1)
	"WAVE1_KNOBS_BLOCK": {"memberKey", "sessionOrder", "capacityLivenessSeconds"},
	// MULTIPOD_KNOBS_BLOCK (EMIT_MP_KNOBS=1)
	"MULTIPOD_KNOBS_BLOCK": {"engineGroups", "bindingMode", "residue", "residueReconcile",
		"engineGate", "satKvThreshold", "satQueueThreshold"},
	// FARM_KNOBS_BLOCK (EMIT_FA_KNOBS=1)
	"FARM_KNOBS_BLOCK": {"quantumMode", "quantumCeiling", "entryOrder", "rotateWhenFree",
		"maxNumBatchedTokens", "longQuantum"},
	// PAT_KNOBS_BLOCK (ELIGIBILITY_MODE=pat)
	"PAT_KNOBS_BLOCK": {"eligibilityMode"},
}

// TestEntryOrderValuesMatchTheEmitter pins the ENUM VALUES the exp-ops renderer can put
// into `entryOrder`. The key-level check above catches a removed FIELD; this catches a
// removed VALUE, which fails just as hard at plugin init (validate() rejects it and the
// EPP CrashLoops on every arm that renders it) but would otherwise only show up at the
// first deploy of the arm that carries it. Keep the list in step with the entryOrder
// comment in exp-ops/lib/common.sh.
func TestEntryOrderValuesMatchTheEmitter(t *testing.T) {
	// the values the emitter documents (FARM_KNOBS_BLOCK / configs/*.env)
	emitted := []string{"newcomer-first", "hrrn", "hrrn-bounded"}
	for _, v := range emitted {
		c := defaultConfig()
		c.EntryOrder = v
		if err := c.validate(); err != nil {
			t.Errorf("exp-ops can render entryOrder=%q but Config.validate rejects it: %v", v, err)
		}
	}
	// ... and an unknown value must still fail closed instead of silently ordering by
	// the default (a typo in an arm's env would otherwise measure the anchor twice).
	c := defaultConfig()
	c.EntryOrder = "hrrn-boundedd"
	if err := c.validate(); err == nil {
		t.Fatal("an unknown entryOrder must be rejected")
	}
}

// TestConfigEmitterConformance fails when the emitter and Config disagree — in either
// direction that can crash a deploy.
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
	for block, keys := range emitterKeys {
		for _, k := range keys {
			if !tags[k] {
				t.Errorf("exp-ops emits %q (%s) but Config has no such field: the EPP rejects unknown fields, so every arm carrying that block CrashLoops at startup. Update exp-ops/lib/common.sh (and this list), or restore the knob.", k, block)
			}
		}
	}
}
