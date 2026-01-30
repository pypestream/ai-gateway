// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package tracing

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/propagators/autoprop"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// metadataTestRecorder is a minimal recorder for testing metadata functionality
type metadataTestRecorder struct{}

func (r metadataTestRecorder) StartParams(_ *openai.ChatCompletionRequest, _ []byte) (string, []oteltrace.SpanStartOption) {
	return "default-span-name", []oteltrace.SpanStartOption{oteltrace.WithSpanKind(oteltrace.SpanKindServer)}
}

func (r metadataTestRecorder) RecordRequest(_ oteltrace.Span, _ *openai.ChatCompletionRequest, _ []byte) {
}

func (r metadataTestRecorder) RecordResponse(_ oteltrace.Span, _ *openai.ChatCompletionResponse) {
}

func (r metadataTestRecorder) RecordResponseChunks(_ oteltrace.Span, _ []*openai.ChatCompletionResponseChunk) {
}

func (r metadataTestRecorder) RecordResponseOnError(_ oteltrace.Span, _ int, _ []byte) {
}

func TestSpanNameOverride(t *testing.T) {
	tests := []struct {
		name             string
		headers          map[string]string
		expectedSpanName string
	}{
		{
			name:             "no override header uses default",
			headers:          map[string]string{},
			expectedSpanName: "default-span-name",
		},
		{
			name: "override header changes span name",
			headers: map[string]string{
				tracingapi.SpanNameHeaderName: "custom-operation",
			},
			expectedSpanName: "custom-operation",
		},
		{
			name: "override with whitespace is trimmed",
			headers: map[string]string{
				tracingapi.SpanNameHeaderName: "  custom-operation  ",
			},
			expectedSpanName: "custom-operation",
		},
		{
			name: "empty override header uses default",
			headers: map[string]string{
				tracingapi.SpanNameHeaderName: "",
			},
			expectedSpanName: "default-span-name",
		},
		{
			name: "whitespace-only override header uses default",
			headers: map[string]string{
				tracingapi.SpanNameHeaderName: "   ",
			},
			expectedSpanName: "default-span-name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up test tracer with in-memory span exporter
			exporter := tracetest.NewInMemoryExporter()
			tp := trace.NewTracerProvider(
				trace.WithSyncer(exporter),
			)
			defer func() { _ = tp.Shutdown(context.Background()) }()

			tracer := tp.Tracer("test")
			propagator := autoprop.NewTextMapPropagator()

			// Create the request tracer
			reqTracer := newChatCompletionTracer(
				tracer,
				propagator,
				metadataTestRecorder{},
				nil,
			)

			// Start a span with the test headers
			ctx := context.Background()
			req := &openai.ChatCompletionRequest{
				Model: openai.ModelGPT5Nano,
			}
			carrier := propagation.MapCarrier{}

			span := reqTracer.StartSpanAndInjectHeaders(ctx, tt.headers, carrier, req, []byte("{}"))
			require.NotNil(t, span)

			// End the span to flush it to the exporter
			span.EndSpan()

			// Verify the span name
			spans := exporter.GetSpans()
			require.Len(t, spans, 1)
			require.Equal(t, tt.expectedSpanName, spans[0].Name)
		})
	}
}

func TestMetadataHeaderParsing(t *testing.T) {
	tests := []struct {
		name              string
		headers           map[string]string
		expectedAttrsMap  map[string]string
		checkAttrContains []string // Attributes that should be present
	}{
		{
			name:              "no metadata header",
			headers:           map[string]string{},
			checkAttrContains: []string{},
		},
		{
			name: "simple metadata",
			headers: map[string]string{
				tracingapi.MetadataHeaderName: `{"user_id":"123","session":"abc"}`,
			},
			expectedAttrsMap: map[string]string{
				"metadata.user_id": "123",
				"metadata.session": "abc",
			},
		},
		{
			name: "nested metadata",
			headers: map[string]string{
				tracingapi.MetadataHeaderName: `{"user":{"id":"123","name":"John"},"session":{"id":"abc"}}`,
			},
			expectedAttrsMap: map[string]string{
				"metadata.user.id":    "123",
				"metadata.user.name":  "John",
				"metadata.session.id": "abc",
			},
		},
		{
			name: "invalid JSON stores as raw",
			headers: map[string]string{
				tracingapi.MetadataHeaderName: `not valid json`,
			},
			expectedAttrsMap: map[string]string{
				"metadata.raw": "not valid json",
			},
		},
		{
			name: "metadata with various types",
			headers: map[string]string{
				tracingapi.MetadataHeaderName: `{"string":"text","number":42,"bool":true,"null":null}`,
			},
			expectedAttrsMap: map[string]string{
				"metadata.string": "text",
				"metadata.number": "42",
				"metadata.bool":   "true",
				"metadata.null":   "",
			},
		},
		{
			name: "metadata with array",
			headers: map[string]string{
				tracingapi.MetadataHeaderName: `{"tags":["tag1","tag2"]}`,
			},
			expectedAttrsMap: map[string]string{
				"metadata.tags.0": "tag1",
				"metadata.tags.1": "tag2",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up test tracer with in-memory span exporter
			exporter := tracetest.NewInMemoryExporter()
			tp := trace.NewTracerProvider(
				trace.WithSyncer(exporter),
			)
			defer func() { _ = tp.Shutdown(context.Background()) }()

			tracer := tp.Tracer("test")
			propagator := autoprop.NewTextMapPropagator()

			// Create the request tracer
			reqTracer := newChatCompletionTracer(
				tracer,
				propagator,
				metadataTestRecorder{},
				nil,
			)

			// Start a span with the test headers
			ctx := context.Background()
			req := &openai.ChatCompletionRequest{
				Model: openai.ModelGPT5Nano,
			}
			carrier := propagation.MapCarrier{}

			span := reqTracer.StartSpanAndInjectHeaders(ctx, tt.headers, carrier, req, []byte("{}"))
			require.NotNil(t, span)

			// End the span to flush it to the exporter
			span.EndSpan()

			// Verify the span attributes
			spans := exporter.GetSpans()
			require.Len(t, spans, 1)

			// Convert span attributes to a map for easier comparison
			attrs := spans[0].Attributes
			attrMap := make(map[string]string)
			for _, attr := range attrs {
				// Only check metadata attributes
				keyStr := string(attr.Key)
				if len(keyStr) >= 9 && keyStr[:9] == "metadata." {
					attrMap[keyStr] = attr.Value.AsString()
				} else if keyStr == "metadata.raw" {
					attrMap[keyStr] = attr.Value.AsString()
				}
			}

			// Check expected attributes
			for key, expectedValue := range tt.expectedAttrsMap {
				actualValue, found := attrMap[key]
				require.True(t, found, "Expected attribute %s not found", key)
				require.Equal(t, expectedValue, actualValue, "Attribute %s has wrong value", key)
			}
		})
	}
}

func TestSpanNameAndMetadataTogether(t *testing.T) {
	// Set up test tracer with in-memory span exporter
	exporter := tracetest.NewInMemoryExporter()
	tp := trace.NewTracerProvider(
		trace.WithSyncer(exporter),
	)
	defer func() { _ = tp.Shutdown(context.Background()) }()

	tracer := tp.Tracer("test")
	propagator := autoprop.NewTextMapPropagator()

	// Create the request tracer
	reqTracer := newChatCompletionTracer(
		tracer,
		propagator,
		testChatCompletionRecorder{},
		nil,
	)

	// Start a span with both custom span name and metadata
	ctx := context.Background()
	headers := map[string]string{
		tracingapi.SpanNameHeaderName: "custom-operation",
		tracingapi.MetadataHeaderName: `{"user_id":"123","tier":"premium"}`,
	}
	req := &openai.ChatCompletionRequest{
		Model: openai.ModelGPT5Nano,
	}
	carrier := propagation.MapCarrier{}

	span := reqTracer.StartSpanAndInjectHeaders(ctx, headers, carrier, req, []byte("{}"))
	require.NotNil(t, span)

	// End the span to flush it to the exporter
	span.EndSpan()

	// Verify both span name and attributes
	spans := exporter.GetSpans()
	require.Len(t, spans, 1)

	// Check span name
	require.Equal(t, "custom-operation", spans[0].Name)

	// Check metadata attributes
	attrs := spans[0].Attributes
	attrMap := make(map[string]string)
	for _, attr := range attrs {
		attrMap[string(attr.Key)] = attr.Value.AsString()
	}

	require.Equal(t, "123", attrMap["metadata.user_id"])
	require.Equal(t, "premium", attrMap["metadata.tier"])
}
