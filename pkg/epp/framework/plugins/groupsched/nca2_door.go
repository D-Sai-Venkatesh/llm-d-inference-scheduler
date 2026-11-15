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

// THE DOOR FIX (session-layer contract Part A, `doorBackfill`): ONE outside thread — the head — may
// collect GPU seats, on ONE pod, and only when no pod can take it right now; every other outsider is
// admitted the moment it fits what is already free (backfill). It replaces ncaV2DecideOnce, and nothing
// else, when the switch is on; with it off (n.door == nil) none of this file runs (C7).

import (
	"fmt"
	"strconv"
)

// nca2Door is the door's configuration and its state across passes (ST3).
type nca2Door struct {
	// sessionAware is Part B; header is the lower-case session header it reads (C1/C4).
	sessionAware bool
	header       string
}

// doorEcho is C6's part of the nca2 config echo: the three resolved values, whether or not they were set.
func (n *nca2Policy) doorEcho() string {
	door, aware, header := false, false, ""
	if n != nil && n.door != nil {
		door, aware, header = true, n.door.sessionAware, n.door.header
	}
	return fmt.Sprintf("doorBackfill=%s sessionAware=%s sessionHeader=%s",
		strconv.FormatBool(door), strconv.FormatBool(aware), header)
}
