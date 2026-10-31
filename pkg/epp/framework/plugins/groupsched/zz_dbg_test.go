package groupsched

import (
	"fmt"
	"os"
	"testing"
	"time"
)

func TestZZTrace(t *testing.T) {
	if os.Getenv("GSDBG") == "" {
		t.Skip("no GSDBG")
	}
	debugSeat = func(s *Scheduler, now time.Time, head *waiter, cand *seatChoice) {
		g := s.groups["engine-a"]
		ne := int64(g.index.UniqueBlocksNELocked())
		fit := "nil"
		if cand != nil {
			fit = fmt.Sprintf("%v free=%d cost=%d", cand.fits, cand.freeBudget, cand.costBlocks)
		}
		fmt.Printf("DBG t=%d seats=%d waiters=%d ne=%d pend=%d budget=%d head=%s since=%s cold=%d cand=%s\n",
			now.Unix()-int64(1793000000), len(g.members), len(s.waiters), ne, g.pendingTotal, g.budgetBlocks,
			head.id, head.since.Format("05"), head.coldTokens, fit)
	}
	debugVictim = func(s *Scheduler, now time.Time, g *engineGroup, reason string, victim *member) {
		t := now.Unix() - 1793000000
		if t < 860 || t > 890 {
			return
		}
		var kinds []string
		for _, m := range g.members {
			r, ok := s.eligibleLocked(m, now)
			kinds = append(kinds, fmt.Sprintf("%s{el=%v/%s wait=%v infl=%d bp=%d}", m.id, ok, r, !m.waitingSince.IsZero(), m.inflight, m.busyPeriods))
		}
		vid := "-"
		if victim != nil {
			vid = victim.id
		}
		fmt.Printf("VICTIM t=%d victim=%s reason=%s exits=%v members=%v\n", t, vid, reason, g.counters.exits, kinds)
	}
	if err := os.Setenv("GSDBG", "1"); err != nil {
		t.Fatal(err)
	}
	_ = runGoldenScenario(t)
}
