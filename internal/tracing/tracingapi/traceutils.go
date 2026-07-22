// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package tracingapi

import (
	"os"
	"strings"
)

// SpanNameHeaderName is the HTTP header name used to override span names.
// Can be configured via AIGW_SPAN_NAME_HEADER_NAME environment variable.
var SpanNameHeaderName = resolveSpanNameHeaderName()

// resolveSpanNameHeaderName resolves the header name for custom span names.
// Defaults to "x-ai-span-name" if AIGW_SPAN_NAME_HEADER_NAME is not set.
func resolveSpanNameHeaderName() string {
	if v := strings.TrimSpace(os.Getenv("AIGW_SPAN_NAME_HEADER_NAME")); v != "" {
		return strings.ToLower(v)
	}
	return "x-ai-span-name"
}
