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

// spanNameTestRecorder is a minimal recorder for testing span name override.
type spanNameTestRecorder struct{}

func (r spanNameTestRecorder) StartParams(_ *openai.ChatCompletionRequest, _ []byte) (string, []oteltrace.SpanStartOption) {
	return "default-span-name", []oteltrace.SpanStartOption{oteltrace.WithSpanKind(oteltrace.SpanKindServer)}
}

func (r spanNameTestRecorder) RecordRequest(_ oteltrace.Span, _ *openai.ChatCompletionRequest, _ []byte) {
}

func (r spanNameTestRecorder) RecordResponse(_ oteltrace.Span, _ *openai.ChatCompletionResponse) {
}

func (r spanNameTestRecorder) RecordResponseChunks(_ oteltrace.Span, _ []*openai.ChatCompletionResponseChunk) {
}

func (r spanNameTestRecorder) RecordResponseOnError(_ oteltrace.Span, _ int, _ []byte) {
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
				spanNameTestRecorder{},
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
