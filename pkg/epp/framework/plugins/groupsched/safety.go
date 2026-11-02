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

// THE LIVE SAMPLE PATH — what survives the safety net's removal.
//
// The seat budget is the whole admission control ("safety: none" is the one value),
// and the memory-only net that used to read a live kv scrape is gone with its two
// consumers (the per-engine holds and the framework detector's pool score). What
// remains here is the CAPACITY acquisition path, which was never about safety: the
// aggregate metric `num_gpu_blocks` can be absent from an endpoint's creation event
// (the event fires before the first scrape), so the policy re-reads the LIVE
// candidate set the framework hands its detector once a second and learns each
// engine's capacity from it. A remembered endpoint handle carries a
// construction-time metrics CLONE, which is why the live set is the only input that
// can see the geometry arrive late (the capacity race), and why the ticker's price
// for it — one map read per engine per second — is the price of the fix.

import (
	"time"
)

// engineSample is one engine's LIVE reading: the KV geometry the capacity path
// learns from, and the one packet it lives in. It is the ONLY metrics view in the
// core (see the file header).
type engineSample struct {
	blockSz int
	numBlk  int
	at      time.Time
}
