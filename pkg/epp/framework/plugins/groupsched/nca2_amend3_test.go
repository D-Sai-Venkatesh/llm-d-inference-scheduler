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

// LOGIC Amendment 3: the cold valve counts a returner's CPU-warm context as warm (valveCPUWarm, C9/V1),
// and seats are released by silence only (noQuantum, C10/Q0).

import (
	"encoding/json"
	"strings"
	"testing"
)

// resolveParams decodes a parameters block strictly and resolves its nca2 policy.
func resolveParams(t *testing.T, params string) *nca2Policy {
	t.Helper()
	cfg := defaultConfig()
	dec := json.NewDecoder(strings.NewReader(params))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		t.Fatal(err)
	}
	n, err := cfg.resolveNCA2()
	if err != nil {
		t.Fatalf("%s must resolve, got %v", params, err)
	}
	return n
}

// TestC9C10_KeysRequireNCA2: valveCPUWarm and noQuantum are nca2 keys: either one without mode: nca2
// refuses the startup, whatever its value (presence is the fact).
func TestC9C10_KeysRequireNCA2(t *testing.T) {
	for name, params := range map[string]string{
		"valveCPUWarm without mode":       `{"valveCPUWarm":true}`,
		"valveCPUWarm false without mode": `{"valveCPUWarm":false}`,
		"noQuantum without mode":          `{"noQuantum":true}`,
		"noQuantum false without mode":    `{"noQuantum":false}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := factoryWith(t, params); err == nil {
				t.Fatalf("%s must refuse the startup", name)
			}
		})
	}
}

// TestC9C10_KeysResolve: each key resolves into the nca2 policy, with the door or without it; absent or
// false is off.
func TestC9C10_KeysResolve(t *testing.T) {
	if n := resolveParams(t, `{"mode":"nca2"}`); n.valveCPUWarm || n.noQuantum {
		t.Fatalf("absent keys are off: %+v", n)
	}
	if n := resolveParams(t, `{"mode":"nca2","valveCPUWarm":false,"noQuantum":false}`); n.valveCPUWarm || n.noQuantum {
		t.Fatalf("false keys are off: %+v", n)
	}
	if n := resolveParams(t, `{"mode":"nca2","valveCPUWarm":true}`); !n.valveCPUWarm || n.noQuantum || n.door != nil {
		t.Fatalf("C9 alone, v2's decision: %+v", n)
	}
	if n := resolveParams(t, `{"mode":"nca2","noQuantum":true}`); n.valveCPUWarm || !n.noQuantum || n.door != nil {
		t.Fatalf("C10 alone, v2's decision: %+v", n)
	}
	n := resolveParams(t, `{"mode":"nca2","doorBackfill":true,"backfillReclaim":true,"valveCPUWarm":true,"noQuantum":true}`)
	if !n.valveCPUWarm || !n.noQuantum || n.door == nil || !n.door.backfillReclaim {
		t.Fatalf("both with the door: %+v", n)
	}
	if err := factoryWith(t, `{"mode":"nca2","valveCPUWarm":true,"noQuantum":true}`); err != nil {
		t.Fatalf("the factory accepts both keys in nca2: %v", err)
	}
}

// TestC9C10_Echo: the nca2 echo prints both resolved values, set or not; the effective-config echo carries
// them when set; OFF's echo names neither.
func TestC9C10_Echo(t *testing.T) {
	cfg := defaultConfig()
	cfg.Mode = modeNCA2
	n, err := cfg.resolveNCA2()
	if err != nil {
		t.Fatal(err)
	}
	if got := n.amend3Echo(); got != "valveCPUWarm=false noQuantum=false" {
		t.Fatalf("both off: %q", got)
	}
	on := true
	cfg.ValveCPUWarm, cfg.NoQuantum = &on, &on
	if n, err = cfg.resolveNCA2(); err != nil {
		t.Fatal(err)
	}
	if got := n.amend3Echo(); got != "valveCPUWarm=true noQuantum=true" {
		t.Fatalf("both on: %q", got)
	}
	if echo := configEcho(cfg); !strings.Contains(echo, "valveCPUWarm=true") || !strings.Contains(echo, "noQuantum=true") {
		t.Fatalf("the effective-config echo carries the keys that were set: %s", echo)
	}
	for _, key := range []string{"valveCPUWarm", "noQuantum"} {
		if strings.Contains(configEcho(defaultConfig()), key) {
			t.Fatalf("the OFF echo must not name %q", key)
		}
	}
}
