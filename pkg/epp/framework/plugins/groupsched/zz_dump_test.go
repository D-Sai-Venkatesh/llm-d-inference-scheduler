package groupsched

import (
	"os"
	"testing"
)

func TestZZDump(t *testing.T) {
	if err := os.WriteFile("/tmp/core-golden.txt", []byte(runGoldenScenario(t)), 0o644); err != nil {
		t.Fatal(err)
	}
}
