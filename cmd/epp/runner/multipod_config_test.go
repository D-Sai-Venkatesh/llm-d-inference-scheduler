package runner

import (
	"context"
	"os"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/branchidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	sigsyaml "sigs.k8s.io/yaml"
)

const mpArmDir = "/Users/sai/.sai-ka-knowledge-bank/explore/ring-window-v2/cluster-prep/multipod"

func TestMultipodArmConfigsParse(t *testing.T) {
	for _, name := range []string{"e3-gsch-perengine-plugins.yaml", "e2-affinity-plugins.yaml"} {
		t.Run(name, func(t *testing.T) {
			text, err := os.ReadFile(mpArmDir + "/" + name)
			require.NoError(t, err, "the prepared multi-pod arm config must exist")
			// the plugins register into controller-runtime's PROCESS-GLOBAL
			// registry, so each arm needs a fresh one (two arms in one test
			// binary would otherwise collide on every metric name).
			prev := ctrlmetrics.Registry
			ctrlmetrics.Registry = prometheus.NewRegistry()
			t.Cleanup(func() { ctrlmetrics.Registry = prev })

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			opts := runserver.NewOptions()
			opts.ConfigText = string(text)
			opts.PoolName = "multipod-dry-run"
			opts.AllowExperimentalPlugins = true
			r := NewRunner()
			raw, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err, "must parse")
			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			_, err = r.parseConfigurationPhaseTwo(ctx, raw, ds, opts.RefreshMetricsInterval)
			require.NoError(t, err, "must instantiate (the plugin DAG/layer gates run here: the wave-1 rejection class)")
		})
	}
}

// TestE2SourceMatchesBranchIdentityKey is the CONFIG-IDENTITY proof behind the
// prereg's silent-no-op hazard: the session-affinity source must address the
// EXACT key branch-identity writes. branch-identity writes NewDataKey(
// "branch-identity", "") (producer name empty), and the session_id strategy
// builds the read key with WithNonEmptyProducerName(source.Producer), which is
// a no-op for an empty producer. So the source must be {attribute} with the
// producer OMITTED; naming one addresses "branch-identity/branch-identity",
// a key nobody writes, and the arm silently passes every request through.
func TestE2SourceMatchesBranchIdentityKey(t *testing.T) {
	raw, err := os.ReadFile(mpArmDir + "/e2-affinity-plugins.yaml")
	require.NoError(t, err)
	var cfg struct {
		Plugins []struct {
			Type       string `json:"type"`
			Name       string `json:"name"`
			Parameters struct {
				Strategy        string `json:"strategy"`
				SessionIDConfig struct {
					Sources []struct {
						Attribute string `json:"attribute"`
						Producer  string `json:"producer"`
						Header    string `json:"header"`
					} `json:"sources"`
					EvictionTTLSeconds float64 `json:"evictionTtlSeconds"`
				} `json:"sessionIdConfig"`
			} `json:"parameters"`
		} `json:"plugins"`
	}
	require.NoError(t, sigsyaml.Unmarshal(raw, &cfg))

	var found bool
	for _, p := range cfg.Plugins {
		if p.Type != sessionaffinity.SessionAffinityType {
			continue
		}
		found = true
		require.Equal(t, sessionaffinity.StrategySessionID, p.Parameters.Strategy)
		require.GreaterOrEqual(t, p.Parameters.SessionIDConfig.EvictionTTLSeconds, 3600.0,
			"the TTL must be >= the measured p99.9 inter-turn idle gap (3,036 s); the plugin default is 300 s and silently unbinds most members")
		require.Len(t, p.Parameters.SessionIDConfig.Sources, 1)
		src := p.Parameters.SessionIDConfig.Sources[0]
		require.Empty(t, src.Header, "the E2 arm has no session header; the id comes from the derived branch attribute")
		require.Equal(t, "branch-identity", src.Attribute)
		// the read key the strategy builds for this source
		readKey := fwkplugin.NewDataKey(src.Attribute, "").WithNonEmptyProducerName(src.Producer)
		require.Equal(t, branchidentity.BranchIdentityKey, readKey,
			"the source must address the key branch-identity WRITES (%q); producer %q addresses %q",
			branchidentity.BranchIdentityKey, src.Producer, readKey)
		// the failure class, stated as an assertion: naming the producer misses
		missed := fwkplugin.NewDataKey(src.Attribute, "").WithNonEmptyProducerName("branch-identity")
		require.NotEqual(t, branchidentity.BranchIdentityKey, missed,
			"a named producer must NOT accidentally match the written key")
	}
	require.True(t, found, "the E2 arm must deploy the session-affinity filter")
	// and no prefixcacheaffinity: its maxTTFTPenaltyMs escape is the knob the
	// prereg requires DISABLED, and it is absent by construction here.
	for _, p := range cfg.Plugins {
		require.NotContains(t, p.Type, "prefix-cache-affinity",
			"the E2 chain must not carry the prefix-cache-affinity filter (its TTFT escape breaks stickiness)")
	}
}

// mpArmConfig is the subset of an EndpointPickerConfig the arm-discipline tests
// read (plugin types in order, the profile list, the flow-control wiring).
type mpArmConfig struct {
	Plugins []struct {
		Type       string `json:"type"`
		Name       string `json:"name"`
		Parameters struct {
			EngineGroups string `json:"engineGroups"`
		} `json:"parameters"`
	} `json:"plugins"`
	SchedulingProfiles []struct {
		Name    string `json:"name"`
		Plugins []struct {
			PluginRef string `json:"pluginRef"`
		} `json:"plugins"`
	} `json:"schedulingProfiles"`
	FlowControl struct {
		PriorityBands []struct {
			FairnessPolicyRef string `json:"fairnessPolicyRef"`
		} `json:"priorityBands"`
	} `json:"flowControl"`
}

func readArm(t *testing.T, name string) mpArmConfig {
	t.Helper()
	raw, err := os.ReadFile(mpArmDir + "/" + name)
	require.NoError(t, err)
	var cfg mpArmConfig
	require.NoError(t, sigsyaml.Unmarshal(raw, &cfg))
	return cfg
}

// TestE2E3OneVariableDelta is the ARM DISCIPLINE guard: E2 and E3 must differ by
// exactly one variable — the group scheduler (plus the fairnessPolicyRef that
// binds it into flow control). Two audit blockers are pinned here:
//
//	B1  E3 must carry the SAME branch-homing filter as E2 (in both `plugins:`
//	    and the scheduling profile, BEFORE the scorer). Without it E3's
//	    placement is argmax(token-load) per request, seated members ping-pong
//	    between engines, every turn unseats + force-seats + re-prefills cold,
//	    and "multi-pod gsch is dominated" would be a false negative.
//	B2  E2 must carry the SAME token-producer as E3, or the two arms compute
//	    their placement inputs from different numbers (auto-created estimate
//	    backend vs the render sidecar).
func TestE2E3OneVariableDelta(t *testing.T) {
	e2 := readArm(t, "e2-affinity-plugins.yaml")
	e3 := readArm(t, "e3-gsch-perengine-plugins.yaml")

	types := func(c mpArmConfig) []string {
		out := make([]string, 0, len(c.Plugins))
		for _, p := range c.Plugins {
			out = append(out, p.Type)
		}
		return out
	}
	t2, t3 := types(e2), types(e3)
	extra := map[string]int{}
	for _, ty := range t3 {
		extra[ty]++
	}
	for _, ty := range t2 {
		extra[ty]--
	}
	for ty, n := range extra {
		if n == 0 {
			continue
		}
		require.Equal(t, 1, n, "plugin %q: E3 has %d more instance(s) than E2 (the arms must differ by ONE variable)", ty, n)
		require.Equal(t, "group-scheduler", ty, "the ONLY plugin E3 may add is the group scheduler")
	}
	// nothing E2-only may exist (the filter must be in BOTH — blocker B1)
	for ty, n := range extra {
		require.GreaterOrEqual(t, n, 0, "plugin %q exists only in E2 — E3 must carry the same identity/homing chain", ty)
	}

	// B1: the homing filter, in `plugins:` AND in the profile, BEFORE the scorer
	profiles := func(c mpArmConfig) []string {
		var out []string
		for _, p := range c.SchedulingProfiles {
			for _, r := range p.Plugins {
				out = append(out, r.PluginRef)
			}
		}
		return out
	}
	require.Equal(t, profiles(e2), profiles(e3),
		"E2 and E3 must run the SAME scheduling profile (filter before scorer)")
	require.Equal(t, []string{"branch-homing", "token-load-scorer"}, profiles(e3),
		"the homing filter must be FIRST in the profile (it narrows the candidate set; the scorer orders the rest)")

	// the fairnessPolicyRef is the ONE flow-control difference
	require.Empty(t, e2.FlowControl.PriorityBands, "E2 runs flow control's default FCFS: no fairness policy")
	require.Len(t, e3.FlowControl.PriorityBands, 1)
	require.Equal(t, "group", e3.FlowControl.PriorityBands[0].FairnessPolicyRef)

	// E3's engine scope
	for _, p := range e3.Plugins {
		if p.Type == "group-scheduler" {
			require.Equal(t, "per-engine", p.Parameters.EngineGroups,
				"E3 is a 2-pod arm: a pooled (single) group would silently measure the wrong thing")
		}
	}
}
