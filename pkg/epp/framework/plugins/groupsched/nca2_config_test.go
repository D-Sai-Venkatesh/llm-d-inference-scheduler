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
