// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package tracingapi

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel/attribute"

	"github.com/envoyproxy/ai-gateway/internal/json"
)

var (
	// SpanNameHeaderName is the HTTP header name used to override span names.
	// Can be configured via AIGW_SPAN_NAME_HEADER_NAME environment variable.
	SpanNameHeaderName = resolveSpanNameHeaderName()
	// MetadataHeaderName is the HTTP header name used to pass custom metadata.
	// Can be configured via AIGW_METADATA_HEADER_NAME environment variable.
	MetadataHeaderName = resolveMetadataHeaderName()
	// metadataAttrPrefix is the prefix used for all metadata attributes in spans.
	// Can be configured via AIGW_METADATA_ATTR_PREFIX environment variable.
	metadataAttrPrefix = resolveMetadataAttrPrefix()
	// maxFlattenedAttrs limits how many flattened attributes to create before
	// falling back to storing the entire JSON as a single attribute.
	// Can be configured via AIGW_MAX_FLATTENED_ATTRIBUTES environment variable.
	maxFlattenedAttrs = resolveMaxFlattenedAttributes()
)

// resolveSpanNameHeaderName resolves the header name for custom span names.
// Defaults to "x-ai-span-name" if AIGW_SPAN_NAME_HEADER_NAME is not set.
func resolveSpanNameHeaderName() string {
	if v := strings.TrimSpace(os.Getenv("AIGW_SPAN_NAME_HEADER_NAME")); v != "" {
		return strings.ToLower(v)
	}
	return "x-ai-span-name"
}

// resolveMetadataHeaderName resolves the header name for custom metadata.
// Defaults to "x-ai-metadata" if AIGW_METADATA_HEADER_NAME is not set.
func resolveMetadataHeaderName() string {
	if v := strings.TrimSpace(os.Getenv("AIGW_METADATA_HEADER_NAME")); v != "" {
		return strings.ToLower(v)
	}
	return "x-ai-metadata"
}

// resolveMetadataAttrPrefix resolves the prefix for metadata span attributes.
// Ensures the prefix ends with a dot. Defaults to "metadata." if AIGW_METADATA_ATTR_PREFIX is not set.
func resolveMetadataAttrPrefix() string {
	if v := strings.TrimSpace(os.Getenv("AIGW_METADATA_ATTR_PREFIX")); v != "" {
		if strings.HasSuffix(v, ".") {
			return v
		}
		return v + "."
	}
	return "metadata."
}

// resolveMaxFlattenedAttributes resolves the maximum number of flattened attributes
// to create from nested JSON before falling back to a single JSON string attribute.
// Defaults to 50 if AIGW_MAX_FLATTENED_ATTRIBUTES is not set or invalid.
func resolveMaxFlattenedAttributes() int {
	if v := strings.TrimSpace(os.Getenv("AIGW_MAX_FLATTENED_ATTRIBUTES")); v != "" {
		if limit, err := strconv.Atoi(v); err == nil && limit > 0 {
			return limit
		}
	}
	return 50
}

// flattenJSON flattens the top-level keys of a JSON object into individual attributes.
// Nested objects and arrays are serialized as a single JSON string attribute rather than
// being expanded into dot-notation keys. For example, {"a": {"b": "c"}, "x": "y"}
// produces {"a": "{\"b\":\"c\"}", "x": "y"}.
func flattenJSON(prefix string, data any, result map[string]string) {
	switch v := data.(type) {
	case map[string]any:
		if prefix != "" {
			// Nested object: store as JSON string instead of flattening further.
			if b, err := json.Marshal(v); err == nil {
				result[prefix] = string(b)
			} else {
				result[prefix] = fmt.Sprintf("%v", v)
			}
			return
		}
		for key, value := range v {
			flattenJSON(key, value, result)
		}
	case []any:
		if prefix != "" {
			// Array: store as JSON string instead of expanding with numeric indices.
			if b, err := json.Marshal(v); err == nil {
				result[prefix] = string(b)
			} else {
				result[prefix] = fmt.Sprintf("%v", v)
			}
			return
		}
		for i, value := range v {
			flattenJSON(fmt.Sprintf("%d", i), value, result)
		}
	case string:
		result[prefix] = v
	case float64:
		result[prefix] = fmt.Sprintf("%v", v)
	case bool:
		result[prefix] = fmt.Sprintf("%v", v)
	case nil:
		result[prefix] = ""
	default:
		result[prefix] = fmt.Sprintf("%v", v)
	}
}

// ParseMetadataHeader parses the metadata header value as JSON and returns span
// attributes. Top-level scalar values are stored individually; nested objects and
// arrays are stored as a single JSON string attribute. If parsing fails, falls back
// to storing the raw string under "metadata.raw". If the number of attributes exceeds
// maxFlattenedAttributes, stores the entire JSON as a single attribute instead.
func ParseMetadataHeader(headerValue string) []attribute.KeyValue {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(headerValue), &parsed); err != nil {
		// Fallback: store raw string if JSON parsing fails.
		return []attribute.KeyValue{
			attribute.String(metadataAttrPrefix+"raw", headerValue),
		}
	}

	flattened := make(map[string]string)
	flattenJSON("", parsed, flattened)

	// If the flattened structure creates too many attributes,
	// store as a single JSON string instead to avoid hitting span attribute limits.
	if len(flattened) > maxFlattenedAttrs {
		return []attribute.KeyValue{
			attribute.String(metadataAttrPrefix+"json", headerValue),
			attribute.Int(metadataAttrPrefix+"flattened_count", len(flattened)),
		}
	}

	attrs := make([]attribute.KeyValue, 0, len(flattened))
	for key, value := range flattened {
		attrs = append(attrs, attribute.String(metadataAttrPrefix+key, value))
	}
	return attrs
}
