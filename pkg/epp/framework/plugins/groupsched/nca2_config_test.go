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

// THE SELECTOR IS FAIL CLOSED (nca2 spec §1): an absent or "" mode is the shipped policy, "nca2"
// is the only mode that changes anything, and every way of asking for half of nca2 — an unknown
// mode, a key without its mode, a value out of range — refuses the startup instead of running
// something nobody configured.

import (
	"encoding/json"
	"strings"
	"testing"
)

// factoryWith decodes a parameters block through the plugin's own strict factory.
func factoryWith(t *testing.T, params string) error {
	t.Helper()
	_, err := Factory("group", json.NewDecoder(strings.NewReader(params)), nil)
	return err
}

// TestOFFResolvesNothing: no mode means no nca2 state at all, and the released-window bound is the
// base's own idleSeconds — the shipped behaviour, byte for byte.
func TestOFFResolvesNothing(t *testing.T) {
	cfg := defaultConfig()
	n2, err := cfg.resolveNCA2()
	if err != nil {
		t.Fatalf("an absent mode must resolve to OFF, got %v", err)
	}
	if n2 != nil {
		t.Fatalf("OFF must resolve to nil (no nca2 state), got %+v", n2)
	}
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if s.nca2 != nil {
		t.Fatal("a scheduler in OFF must carry no nca2 policy")
	}
	if s.lostWindow != 20*1e9 {
		t.Fatalf("OFF's released-window bound must be idleSeconds (20s), got %v", s.lostWindow)
	}
	if err := factoryWith(t, `{}`); err != nil {
		t.Fatalf("an empty parameters block is the shipped policy, got %v", err)
	}
	if err := factoryWith(t, `{"mode": ""}`); err != nil {
		t.Fatalf("mode: \"\" is the shipped policy, got %v", err)
	}
}

// TestNCA2DefaultsAndEcho: mode: nca2 alone runs the spec's own numbers, and the echo carries them.
func TestNCA2DefaultsAndEcho(t *testing.T) {
	cfg := Config{}
	cfg.Mode = modeNCA2
	cfg.BlockSizeTokens = 64
	cfg.Quantum, cfg.IdleSeconds, cfg.InFlightCap = 4, 5, 64
	cfg.InflightReapSeconds = defaultInflightReapSeconds
	cfg.GroupBudgetFraction = 0.95
	n2, err := cfg.resolveNCA2()
	if err != nil {
		t.Fatal(err)
	}
	if n2.CPUTierTokens != defaultCPUTierTokens {
		t.Fatalf("C must default to %d, got %d", defaultCPUTierTokens, n2.CPUTierTokens)
	}
	tier := defaultCPUTierTokens
	if want := int64(defaultAdmitLine * float64(tier)); n2.AdmitLineTokens != want {
		t.Fatalf("L_a must default to 0.95 C = %d, got %d", want, n2.AdmitLineTokens)
	}
	if n2.TrustLineTokens != n2.AdmitLineTokens {
		t.Fatalf("both lines default to the same 0.95 C: %d vs %d", n2.TrustLineTokens, n2.AdmitLineTokens)
	}
	if n2.MaxInflightSeconds != defaultMaxInflightSeconds || n2.LostRequestSeconds != defaultLostRequestSeconds {
		t.Fatalf("the two time bounds must default to %v / %v, got %v / %v",
			defaultMaxInflightSeconds, defaultLostRequestSeconds, n2.MaxInflightSeconds, n2.LostRequestSeconds)
	}
	s, err := NewScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if s.lostWindow != n2.lostRequestWindow() {
		t.Fatalf("nca2's released-window bound must be lostRequestSeconds, got %v", s.lostWindow)
	}
	// the echo omits a nil key and prints a set one: the OFF echo cannot gain a line
	echo := configEcho(defaultConfig())
	for _, key := range []string{"mode", "cpuTierTokens", "admitLine", "trustLine", "maxInflightSeconds", "lostRequestSeconds"} {
		if strings.Contains(echo, key) {
			t.Fatalf("the OFF echo must not name %q: %s", key, echo)
		}
	}
	if err := factoryWith(t, `{"mode":"nca2","cpuTierTokens":1000,"admitLine":0.5}`); err != nil {
		t.Fatalf("a complete nca2 parameters block must decode, got %v", err)
	}
}

// TestFailClosedOnTheSelector: every half-configured nca2 refuses the startup.
func TestFailClosedOnTheSelector(t *testing.T) {
	cases := map[string]string{
		"unknown mode":               `{"mode":"nca3"}`,
		"cpuTierTokens without mode": `{"cpuTierTokens":1000}`,
		"admitLine without mode":     `{"admitLine":0.5}`,
		"trustLine without mode":     `{"trustLine":0.5}`,
		"maxInflight without mode":   `{"maxInflightSeconds":60}`,
		"lostRequest without mode":   `{"lostRequestSeconds":60}`,
		"zero tier":                  `{"mode":"nca2","cpuTierTokens":0}`,
		"admitLine > 1":              `{"mode":"nca2","admitLine":1.5}`,
		"trustLine 0":                `{"mode":"nca2","trustLine":0}`,
		"zero maxInflight":           `{"mode":"nca2","maxInflightSeconds":0}`,
		"zero lostRequest":           `{"mode":"nca2","lostRequestSeconds":0}`,
		"unknown key":                `{"mode":"nca2","cpuTier":"x"}`,
	}
	for name, params := range cases {
		t.Run(name, func(t *testing.T) {
			if err := factoryWith(t, params); err == nil {
				t.Fatalf("%s must refuse the startup", name)
			}
		})
	}
}

// TestC2_DoorKeysRequireNCA2: the three door keys are nca2 keys — any of them without mode: nca2 refuses
// the startup, whatever its value (presence is the fact).
func TestC2_DoorKeysRequireNCA2(t *testing.T) {
	for name, params := range map[string]string{
		"doorBackfill without mode":       `{"doorBackfill":true}`,
		"doorBackfill false without mode": `{"doorBackfill":false}`,
		"sessionAware without mode":       `{"sessionAware":true}`,
		"sessionHeader without mode":      `{"sessionHeader":"x-llm-d-session-id"}`,
		"backfillReclaim without mode":    `{"backfillReclaim":true}`,
		"backfillReclaim false w/o mode":  `{"backfillReclaim":false}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := factoryWith(t, params); err == nil {
				t.Fatalf("%s must refuse the startup", name)
			}
		})
	}
}

// TestC3toC5_SessionKeysFailClosed: sessionAware needs doorBackfill (C3) and a usable header (C4); a
// header without sessionAware does nothing and is refused (C5).
func TestC3toC5_SessionKeysFailClosed(t *testing.T) {
	for name, params := range map[string]string{
		"C3 sessionAware without doorBackfill":     `{"mode":"nca2","sessionAware":true,"sessionHeader":"x-llm-d-session-id"}`,
		"C3 sessionAware with doorBackfill off":    `{"mode":"nca2","doorBackfill":false,"sessionAware":true,"sessionHeader":"x-llm-d-session-id"}`,
		"C4 no header":                             `{"mode":"nca2","doorBackfill":true,"sessionAware":true}`,
		"C4 empty header":                          `{"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":""}`,
		"C4 the fairness header":                   `{"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":"X-LLM-D-Inference-Fairness-ID"}`,
		"C4 the deprecated fairness alias":         `{"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":"x-gateway-inference-fairness-id"}`,
		"C4 x-request-id":                          `{"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":"X-Request-Id"}`,
		"C4 x-correlation-id":                      `{"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":"x-correlation-id"}`,
		"C5 header without sessionAware":           `{"mode":"nca2","doorBackfill":true,"sessionHeader":"x-llm-d-session-id"}`,
		"C5 header with sessionAware false":        `{"mode":"nca2","doorBackfill":true,"sessionAware":false,"sessionHeader":"x-llm-d-session-id"}`,
		"C8 backfillReclaim without doorBackfill":  `{"mode":"nca2","backfillReclaim":true}`,
		"C8 backfillReclaim with doorBackfill off": `{"mode":"nca2","doorBackfill":false,"backfillReclaim":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := factoryWith(t, params); err == nil {
				t.Fatalf("%s must refuse the startup", name)
			}
		})
	}
}

// TestC1_DoorKeysResolve: the accepted shapes resolve into the policy's door section, and both switches
// off resolve to NO door section at all (C7: the v2 path runs, the new code is not entered).
func TestC1_DoorKeysResolve(t *testing.T) {
	resolve := func(t *testing.T, params string) *nca2Policy {
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
	if n := resolve(t, `{"mode":"nca2"}`); n.door != nil {
		t.Fatal("C7: no door key resolves to no door section")
	}
	if n := resolve(t, `{"mode":"nca2","doorBackfill":false}`); n.door != nil {
		t.Fatal("C7: doorBackfill false resolves to no door section")
	}
	n := resolve(t, `{"mode":"nca2","doorBackfill":true}`)
	if n.door == nil || n.door.sessionAware || n.door.header != "" || n.door.backfillReclaim {
		t.Fatalf("Part A alone: %+v", n.door)
	}
	// C8 (LOGIC Amendment 2): backfillReclaim rides on the door; false is the door at a2142ddc (I14)
	if n := resolve(t, `{"mode":"nca2","doorBackfill":true,"backfillReclaim":true}`); n.door == nil || !n.door.backfillReclaim {
		t.Fatalf("C8: backfillReclaim resolves into the door: %+v", n.door)
	}
	if n := resolve(t, `{"mode":"nca2","doorBackfill":true,"backfillReclaim":false}`); n.door == nil || n.door.backfillReclaim {
		t.Fatalf("C8: backfillReclaim false: %+v", n.door)
	}
	if n := resolve(t, `{"mode":"nca2","backfillReclaim":false}`); n.door != nil {
		t.Fatal("C7/C8: backfillReclaim false without the door resolves to no door section")
	}
	n = resolve(t, `{"mode":"nca2","doorBackfill":true,"sessionAware":true,"sessionHeader":"X-LLM-D-Session-Id"}`)
	if n.door == nil || !n.door.sessionAware || n.door.header != "x-llm-d-session-id" {
		t.Fatalf("Part A + B: the header is kept lower-case (the EPP's header map is): %+v", n.door)
	}
}

// TestC6_EchoNamesTheDoorSwitches: the nca2 echo prints the three resolved values, set or not, so a
// run's read-back can quote them; OFF's echo names none of them.
func TestC6_EchoNamesTheDoorSwitches(t *testing.T) {
	cfg := defaultConfig()
	cfg.Mode = modeNCA2
	n, err := cfg.resolveNCA2()
	if err != nil {
		t.Fatal(err)
	}
	if got := n.doorEcho(); got != "doorBackfill=false sessionAware=false sessionHeader= backfillReclaim=false" {
		t.Fatalf("both off: %q", got)
	}
	on, off := true, defaultConfig()
	cfg.DoorBackfill, cfg.SessionAware, cfg.BackfillReclaim = &on, &on, &on
	header := "x-llm-d-session-id"
	cfg.SessionHeader = &header
	if n, err = cfg.resolveNCA2(); err != nil {
		t.Fatal(err)
	}
	if got := n.doorEcho(); got != "doorBackfill=true sessionAware=true sessionHeader=x-llm-d-session-id backfillReclaim=true" {
		t.Fatalf("both on: %q", got)
	}
	if echo := configEcho(cfg); !strings.Contains(echo, "doorBackfill=true") || !strings.Contains(echo, "sessionHeader=x-llm-d-session-id") ||
		!strings.Contains(echo, "backfillReclaim=true") {
		t.Fatalf("the effective-config echo carries the keys that were set: %s", echo)
	}
	for _, key := range []string{"doorBackfill", "sessionAware", "sessionHeader", "backfillReclaim"} {
		if strings.Contains(configEcho(off), key) {
			t.Fatalf("the OFF echo must not name %q", key)
		}
	}
}
