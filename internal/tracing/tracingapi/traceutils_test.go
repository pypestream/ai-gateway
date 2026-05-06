// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package tracingapi

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
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

func TestResolveMetadataHeaderName(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected string
	}{
		{
			name:     "default value when env var not set",
			envValue: "",
			expected: "x-ai-metadata",
		},
		{
			name:     "custom value from env var",
			envValue: "X-Custom-Metadata",
			expected: "x-custom-metadata",
		},
		{
			name:     "whitespace trimmed",
			envValue: "  X-Meta  ",
			expected: "x-meta",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				t.Setenv("AIGW_METADATA_HEADER_NAME", tt.envValue)
			} else {
				os.Unsetenv("AIGW_METADATA_HEADER_NAME")
			}
			result := resolveMetadataHeaderName()
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestResolveMetadataAttrPrefix(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected string
	}{
		{
			name:     "default value when env var not set",
			envValue: "",
			expected: "metadata.",
		},
		{
			name:     "custom value with trailing dot",
			envValue: "custom.",
			expected: "custom.",
		},
		{
			name:     "custom value without trailing dot adds it",
			envValue: "custom",
			expected: "custom.",
		},
		{
			name:     "whitespace trimmed and dot added",
			envValue: "  myprefix  ",
			expected: "myprefix.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				t.Setenv("AIGW_METADATA_ATTR_PREFIX", tt.envValue)
			} else {
				os.Unsetenv("AIGW_METADATA_ATTR_PREFIX")
			}
			result := resolveMetadataAttrPrefix()
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestResolveMaxFlattenedAttributes(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected int
	}{
		{
			name:     "default value when env var not set",
			envValue: "",
			expected: 50,
		},
		{
			name:     "custom value from env var",
			envValue: "100",
			expected: 100,
		},
		{
			name:     "invalid value falls back to default",
			envValue: "not-a-number",
			expected: 50,
		},
		{
			name:     "zero value falls back to default",
			envValue: "0",
			expected: 50,
		},
		{
			name:     "negative value falls back to default",
			envValue: "-10",
			expected: 50,
		},
		{
			name:     "whitespace trimmed",
			envValue: "  75  ",
			expected: 75,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				t.Setenv("AIGW_MAX_FLATTENED_ATTRIBUTES", tt.envValue)
			} else {
				os.Unsetenv("AIGW_MAX_FLATTENED_ATTRIBUTES")
			}
			result := resolveMaxFlattenedAttributes()
			require.Equal(t, tt.expected, result)
		})
	}
}

// assertFlattenMap compares expected and actual map[string]string. Values that
// look like JSON objects or arrays are compared semantically via require.JSONEq
// so key ordering in the serialized JSON does not matter.
func assertFlattenMap(t *testing.T, expected, actual map[string]string) {
	t.Helper()
	require.Len(t, actual, len(expected), "result map length mismatch")
	for k, expVal := range expected {
		actVal, ok := actual[k]
		require.True(t, ok, "missing key %q in result", k)
		if len(expVal) > 0 && (expVal[0] == '{' || expVal[0] == '[') {
			require.JSONEq(t, expVal, actVal, "JSON mismatch for key %q", k)
		} else {
			require.Equal(t, expVal, actVal, "value mismatch for key %q", k)
		}
	}
}

func TestFlattenJSON(t *testing.T) {
	tests := []struct {
		name     string
		prefix   string
		data     any
		expected map[string]string
	}{
		{
			name:   "simple string value",
			prefix: "",
			data:   "hello",
			expected: map[string]string{
				"": "hello",
			},
		},
		{
			name:   "simple object with scalar values",
			prefix: "",
			data: map[string]any{
				"key1": "value1",
				"key2": "value2",
			},
			expected: map[string]string{
				"key1": "value1",
				"key2": "value2",
			},
		},
		{
			name:   "nested object stored as single JSON string",
			prefix: "",
			data: map[string]any{
				"user": map[string]any{
					"id":   "123",
					"name": "John",
				},
			},
			expected: map[string]string{
				"user": `{"id":"123","name":"John"}`,
			},
		},
		{
			name:   "deeply nested object stored as single JSON string",
			prefix: "",
			data: map[string]any{
				"level1": map[string]any{
					"level2": map[string]any{
						"level3": "value",
					},
				},
			},
			expected: map[string]string{
				"level1": `{"level2":{"level3":"value"}}`,
			},
		},
		{
			name:   "array stored as single JSON string",
			prefix: "",
			data: map[string]any{
				"items": []any{"item1", "item2", "item3"},
			},
			expected: map[string]string{
				"items": `["item1","item2","item3"]`,
			},
		},
		{
			name:   "array with objects stored as single JSON string",
			prefix: "",
			data: map[string]any{
				"users": []any{
					map[string]any{"id": "1", "name": "Alice"},
					map[string]any{"id": "2", "name": "Bob"},
				},
			},
			expected: map[string]string{
				"users": `[{"id":"1","name":"Alice"},{"id":"2","name":"Bob"}]`,
			},
		},
		{
			name:   "numeric values",
			prefix: "",
			data: map[string]any{
				"count":   float64(42),
				"average": float64(3.14),
			},
			expected: map[string]string{
				"count":   "42",
				"average": "3.14",
			},
		},
		{
			name:   "boolean values",
			prefix: "",
			data: map[string]any{
				"active":  true,
				"deleted": false,
			},
			expected: map[string]string{
				"active":  "true",
				"deleted": "false",
			},
		},
		{
			name:   "nil value",
			prefix: "",
			data: map[string]any{
				"empty": nil,
			},
			expected: map[string]string{
				"empty": "",
			},
		},
		{
			name:   "mixed types with nested object as JSON string",
			prefix: "",
			data: map[string]any{
				"string": "text",
				"number": float64(123),
				"bool":   true,
				"null":   nil,
				"nested": map[string]any{
					"key": "value",
				},
			},
			expected: map[string]string{
				"string": "text",
				"number": "123",
				"bool":   "true",
				"null":   "",
				"nested": `{"key":"value"}`,
			},
		},
		{
			name:   "with non-empty prefix stores object as JSON string",
			prefix: "prefix",
			data: map[string]any{
				"key": "value",
			},
			expected: map[string]string{
				"prefix": `{"key":"value"}`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := make(map[string]string)
			flattenJSON(tt.prefix, tt.data, result)
			assertFlattenMap(t, tt.expected, result)
		})
	}
}

// assertAttributesWithJSONValues checks that result contains exactly the expected
// attributes. For attribute values that look like JSON, it does a semantic comparison
// so key ordering in the serialized JSON does not matter.
func assertAttributesWithJSONValues(t *testing.T, expected, result []attribute.KeyValue) {
	t.Helper()
	require.Len(t, result, len(expected), "attribute count mismatch")
	expByKey := make(map[attribute.Key]attribute.Value, len(expected))
	for _, kv := range expected {
		expByKey[kv.Key] = kv.Value
	}
	for _, kv := range result {
		expVal, ok := expByKey[kv.Key]
		require.True(t, ok, "unexpected attribute key %q", kv.Key)
		actStr := kv.Value.AsString()
		expStr := expVal.AsString()
		if len(expStr) > 0 && (expStr[0] == '{' || expStr[0] == '[') {
			require.JSONEq(t, expStr, actStr, "JSON mismatch for attribute %q", kv.Key)
		} else {
			require.Equal(t, expStr, actStr, "value mismatch for attribute %q", kv.Key)
		}
	}
}

func TestParseMetadataHeader(t *testing.T) {
	// Save original values and restore after tests.
	originalPrefix := metadataAttrPrefix
	originalMaxAttrs := maxFlattenedAttrs
	defer func() {
		metadataAttrPrefix = originalPrefix
		maxFlattenedAttrs = originalMaxAttrs
	}()

	metadataAttrPrefix = "metadata."
	maxFlattenedAttrs = 50

	tests := []struct {
		name        string
		headerValue string
		expected    []attribute.KeyValue
	}{
		{
			name:        "valid simple JSON with scalar values",
			headerValue: `{"user_id":"123","session":"abc"}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.user_id", "123"),
				attribute.String("metadata.session", "abc"),
			},
		},
		{
			name:        "valid nested JSON stores objects as single JSON string attributes",
			headerValue: `{"user":{"id":"123","name":"John"},"session":{"id":"abc"}}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.user", `{"id":"123","name":"John"}`),
				attribute.String("metadata.session", `{"id":"abc"}`),
			},
		},
		{
			name:        "invalid JSON falls back to raw",
			headerValue: `not valid json`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.raw", "not valid json"),
			},
		},
		{
			name:        "empty JSON object",
			headerValue: `{}`,
			expected:    []attribute.KeyValue{},
		},
		{
			name:        "JSON with various scalar types",
			headerValue: `{"string":"text","number":123,"bool":true,"null":null}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.string", "text"),
				attribute.String("metadata.number", "123"),
				attribute.String("metadata.bool", "true"),
				attribute.String("metadata.null", ""),
			},
		},
		{
			name:        "JSON with array stored as single JSON string",
			headerValue: `{"tags":["tag1","tag2","tag3"]}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.tags", `["tag1","tag2","tag3"]`),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ParseMetadataHeader(tt.headerValue)
			assertAttributesWithJSONValues(t, tt.expected, result)
		})
	}
}

func TestParseMetadataHeaderWithTooManyAttributes(t *testing.T) {
	// Save original values and restore after test.
	originalPrefix := metadataAttrPrefix
	originalMaxAttrs := maxFlattenedAttrs
	defer func() {
		metadataAttrPrefix = originalPrefix
		maxFlattenedAttrs = originalMaxAttrs
	}()

	metadataAttrPrefix = "metadata."
	maxFlattenedAttrs = 5 // Set a low limit for testing.

	// Create JSON with more than 5 top-level attributes.
	headerValue := `{"a":"1","b":"2","c":"3","d":"4","e":"5","f":"6"}`

	result := ParseMetadataHeader(headerValue)

	// Should have 2 attributes: the JSON string and the count.
	require.Len(t, result, 2)

	var foundJSON, foundCount bool
	for _, attr := range result {
		if attr.Key == "metadata.json" {
			foundJSON = true
			require.Equal(t, headerValue, attr.Value.AsString())
		}
		if attr.Key == "metadata.flattened_count" {
			foundCount = true
			require.Equal(t, int64(6), attr.Value.AsInt64())
		}
	}

	require.True(t, foundJSON, "Expected to find metadata.json attribute")
	require.True(t, foundCount, "Expected to find metadata.flattened_count attribute")
}
