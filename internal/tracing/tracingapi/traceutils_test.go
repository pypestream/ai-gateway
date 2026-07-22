// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package tracingapi

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestResolveSpanNameHeaderName(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected string
	}{
		{
			name:     "default value when env var not set",
			envValue: "",
			expected: "x-ai-span-name",
		},
		{
			name:     "custom value from env var",
			envValue: "X-Custom-Span-Name",
			expected: "x-custom-span-name",
		},
		{
			name:     "whitespace trimmed",
			envValue: "  X-Span-Override  ",
			expected: "x-span-override",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				t.Setenv("AIGW_SPAN_NAME_HEADER_NAME", tt.envValue)
			} else {
				os.Unsetenv("AIGW_SPAN_NAME_HEADER_NAME")
			}
			result := resolveSpanNameHeaderName()
			require.Equal(t, tt.expected, result)
		})
	}
}
