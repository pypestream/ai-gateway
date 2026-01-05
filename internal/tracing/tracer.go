// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package tracing

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	cohereschema "github.com/envoyproxy/ai-gateway/internal/apischema/cohere"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	tracing "github.com/envoyproxy/ai-gateway/internal/tracing/api"
)

const (
	// defaultSpanNameHeaderName is the header that overrides the span name when provided.
	defaultSpanNameHeaderName = "x-ai-span-name"
	// defaultMetadataHeaderName is the header that contains JSON metadata to be parsed
	// and added as flattened span attributes with the "metadata." prefix.
	defaultMetadataHeaderName = "x-ai-metadata"
	// defaultMetadataAttrPrefix is the prefix used for all metadata span attributes.
	defaultMetadataAttrPrefix = "metadata."
	// defaultMaxFlattenedAttributes is the default limit for flattened metadata attributes.
	// If flattening would create more attributes than this, we fall back to storing
	// the metadata as a single JSON string to avoid hitting span attribute limits.
	defaultMaxFlattenedAttributes = 50
)

var (
	// spanNameHeaderName holds the header name to override span names, derived from env once at init.
	spanNameHeaderName = resolveSpanNameHeaderName()
	// metadataHeaderName holds the effective header name to parse, derived from
	// the environment variable once at init.
	metadataHeaderName = resolveMetadataHeaderName()
	// metadataAttrPrefix holds the effective attribute prefix for metadata attributes.
	metadataAttrPrefix = resolveMetadataAttrPrefix()
	// maxFlattenedAttributes holds the maximum number of flattened attributes allowed,
	// derived from the environment variable once at init.
	maxFlattenedAttributes = resolveMaxFlattenedAttributes()
)

// resolveSpanNameHeaderName derives the span name override header from env with a fallback.
func resolveSpanNameHeaderName() string {
	if v := strings.TrimSpace(os.Getenv("AIGW_SPAN_NAME_HEADER_NAME")); v != "" {
		return strings.ToLower(v)
	}
	return defaultSpanNameHeaderName
}

// resolveMetadataHeaderName derives the metadata header name from env with a fallback.
func resolveMetadataHeaderName() string {
	if v := strings.TrimSpace(os.Getenv("AIGW_METADATA_HEADER_NAME")); v != "" {
		return strings.ToLower(v)
	}
	return defaultMetadataHeaderName
}

// resolveMetadataAttrPrefix derives the metadata attribute prefix from env with a fallback.
// Ensures the prefix always ends with a dot for consistent attribute keys.
func resolveMetadataAttrPrefix() string {
	if v := strings.TrimSpace(os.Getenv("AIGW_METADATA_ATTR_PREFIX")); v != "" {
		if strings.HasSuffix(v, ".") {
			return v
		}
		return v + "."
	}
	return defaultMetadataAttrPrefix
}

// resolveMaxFlattenedAttributes derives the max flattened attributes limit from env with a fallback.
func resolveMaxFlattenedAttributes() int {
	if v := strings.TrimSpace(os.Getenv("AIGW_MAX_FLATTENED_ATTRIBUTES")); v != "" {
		if limit, err := strconv.Atoi(v); err == nil && limit > 0 {
			return limit
		}
	}
	return defaultMaxFlattenedAttributes
}

// flattenJSON recursively flattens a nested JSON structure into dot-notation keys.
// For example, {"a": {"b": "c"}} becomes {"a.b": "c"}.
// Arrays are handled by using numeric indices: {"arr": [1,2]} becomes {"arr.0": "1", "arr.1": "2"}.
func flattenJSON(prefix string, data any, result map[string]string) {
	switch v := data.(type) {
	case map[string]any:
		for key, value := range v {
			newKey := key
			if prefix != "" {
				newKey = prefix + "." + key
			}
			flattenJSON(newKey, value, result)
		}
	case []any:
		for i, value := range v {
			newKey := fmt.Sprintf("%s.%d", prefix, i)
			if prefix == "" {
				newKey = fmt.Sprintf("%d", i)
			}
			flattenJSON(newKey, value, result)
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

// parseMetadataHeader attempts to parse the metadata header value as JSON and
// returns flattened span attributes. If parsing fails, it falls back to storing
// the raw string value under "metadata.raw".
// If flattening would create more than maxFlattenedAttributes, it stores the
// entire JSON as a single attribute to avoid hitting span attribute limits.
func parseMetadataHeader(headerValue string) []attribute.KeyValue {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(headerValue), &parsed); err != nil {
		// Fallback: store raw string if JSON parsing fails.
		return []attribute.KeyValue{
			attribute.String(metadataAttrPrefix+"raw", headerValue),
		}
	}

	flattened := make(map[string]string)
	flattenJSON("", parsed, flattened)

	// If the flattened structure would create too many attributes,
	// store as a single JSON string instead to avoid hitting span attribute limits.
	if len(flattened) > maxFlattenedAttributes {
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

// spanFactory is a function type that creates a new SpanT given a trace.Span and a Recorder.
type spanFactory[ReqT any, RespT any, RespChunkT any] func(trace.Span, tracing.SpanRecorder[ReqT, RespT, RespChunkT]) tracing.Span[RespT, RespChunkT]

// requestTracerImpl implements RequestTracer for various request and span types.
type requestTracerImpl[ReqT any, RespT any, RespChunkT any] struct {
	tracer           trace.Tracer
	propagator       propagation.TextMapPropagator
	recorder         tracing.SpanRecorder[ReqT, RespT, RespChunkT]
	headerAttributes map[string]string
	newSpan          spanFactory[ReqT, RespT, RespChunkT]
}

var (
	_ tracing.ChatCompletionTracer  = (*chatCompletionTracer)(nil)
	_ tracing.EmbeddingsTracer      = (*embeddingsTracer)(nil)
	_ tracing.CompletionTracer      = (*completionTracer)(nil)
	_ tracing.ImageGenerationTracer = (*imageGenerationTracer)(nil)
	_ tracing.ResponsesTracer       = (*responsesTracer)(nil)
	_ tracing.RerankTracer          = (*rerankTracer)(nil)
)

type (
	chatCompletionTracer  = requestTracerImpl[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]
	embeddingsTracer      = requestTracerImpl[openai.EmbeddingRequest, openai.EmbeddingResponse, struct{}]
	completionTracer      = requestTracerImpl[openai.CompletionRequest, openai.CompletionResponse, openai.CompletionResponse]
	imageGenerationTracer = requestTracerImpl[openai.ImageGenerationRequest, openai.ImageGenerationResponse, struct{}]
	responsesTracer       = requestTracerImpl[openai.ResponseRequest, openai.Response, openai.ResponseStreamEventUnion]
	rerankTracer          = requestTracerImpl[cohereschema.RerankV2Request, cohereschema.RerankV2Response, struct{}]
)

func newRequestTracer[ReqT any, RespT any, RespChunkT any](
	tracer trace.Tracer,
	propagator propagation.TextMapPropagator,
	recorder tracing.SpanRecorder[ReqT, RespT, RespChunkT],
	headerAttributes map[string]string,
	newSpan spanFactory[ReqT, RespT, RespChunkT],
) tracing.RequestTracer[ReqT, RespT, RespChunkT] {
	if _, ok := tracer.(noop.Tracer); ok {
		return tracing.NoopTracer[ReqT, RespT, RespChunkT]{}
	}
	return &requestTracerImpl[ReqT, RespT, RespChunkT]{
		tracer:           tracer,
		propagator:       propagator,
		recorder:         recorder,
		headerAttributes: headerAttributes,
		newSpan:          newSpan,
	}
}

func (t *requestTracerImpl[ReqT, RespT, ChunkT]) StartSpanAndInjectHeaders(
	ctx context.Context,
	headers map[string]string,
	carrier propagation.TextMapCarrier,
	req *ReqT,
	body []byte,
) tracing.Span[RespT, ChunkT] {
	parentCtx := t.propagator.Extract(ctx, propagation.MapCarrier(headers))
	spanName, opts := t.recorder.StartParams(req, body)
	if override, ok := headers[spanNameHeaderName]; ok {
		if name := strings.TrimSpace(override); name != "" {
			spanName = name
		}
	}
	newCtx, span := t.tracer.Start(parentCtx, spanName, opts...)

	t.propagator.Inject(newCtx, carrier)

	var zero tracing.Span[RespT, ChunkT]
	if !span.IsRecording() {
		return zero
	}

	t.recorder.RecordRequest(span, req, body)

	if len(t.headerAttributes) > 0 {
		attrs := make([]attribute.KeyValue, 0, len(t.headerAttributes))
		for headerName, attrName := range t.headerAttributes {
			if headerValue, ok := headers[headerName]; ok {
				attrs = append(attrs, attribute.String(attrName, headerValue))
			}
		}
		if len(attrs) > 0 {
			span.SetAttributes(attrs...)
		}
	}

	// Process the metadata header specially: parse JSON and flatten into span attributes.
	if metadataValue, ok := headers[metadataHeaderName]; ok {
		metadataAttrs := parseMetadataHeader(metadataValue)
		if len(metadataAttrs) > 0 {
			span.SetAttributes(metadataAttrs...)
		}
	}

	return t.newSpan(span, t.recorder)
}

func newChatCompletionTracer(tracer trace.Tracer, propagator propagation.TextMapPropagator, recorder tracing.ChatCompletionRecorder, headerAttributes map[string]string) tracing.ChatCompletionTracer {
	return newRequestTracer(
		tracer,
		propagator,
		recorder,
		headerAttributes,
		func(span trace.Span, recorder tracing.ChatCompletionRecorder) tracing.ChatCompletionSpan {
			return &chatCompletionSpan{span: span, recorder: recorder}
		},
	)
}

func newEmbeddingsTracer(tracer trace.Tracer, propagator propagation.TextMapPropagator, recorder tracing.EmbeddingsRecorder, headerAttributes map[string]string) tracing.EmbeddingsTracer {
	return newRequestTracer(
		tracer,
		propagator,
		recorder,
		headerAttributes,
		func(span trace.Span, recorder tracing.EmbeddingsRecorder) tracing.EmbeddingsSpan {
			return &embeddingsSpan{span: span, recorder: recorder}
		},
	)
}

func newCompletionTracer(tracer trace.Tracer, propagator propagation.TextMapPropagator, recorder tracing.CompletionRecorder, headerAttributes map[string]string) tracing.CompletionTracer {
	return newRequestTracer(
		tracer,
		propagator,
		recorder,
		headerAttributes,
		func(span trace.Span, recorder tracing.CompletionRecorder) tracing.CompletionSpan {
			return &completionSpan{span: span, recorder: recorder}
		},
	)
}

func newImageGenerationTracer(tracer trace.Tracer, propagator propagation.TextMapPropagator, recorder tracing.ImageGenerationRecorder) tracing.ImageGenerationTracer {
	return newRequestTracer(
		tracer,
		propagator,
		recorder,
		nil,
		func(span trace.Span, recorder tracing.ImageGenerationRecorder) tracing.ImageGenerationSpan {
			return &imageGenerationSpan{span: span, recorder: recorder}
		},
	)
}

func newResponsesTracer(tracer trace.Tracer, propagator propagation.TextMapPropagator, recorder tracing.ResponsesRecorder, headerAttributes map[string]string) tracing.ResponsesTracer {
	return newRequestTracer(
		tracer,
		propagator,
		recorder,
		headerAttributes,
		func(span trace.Span, recorder tracing.ResponsesRecorder) tracing.ResponsesSpan {
			return &responsesSpan{span: span, recorder: recorder}
		},
	)
}

func newRerankTracer(tracer trace.Tracer, propagator propagation.TextMapPropagator, recorder tracing.RerankRecorder, headerAttributes map[string]string) tracing.RerankTracer {
	return newRequestTracer(
		tracer,
		propagator,
		recorder,
		headerAttributes,
		func(span trace.Span, recorder tracing.RerankRecorder) tracing.RerankSpan {
			return &rerankSpan{span: span, recorder: recorder}
		},
	)
}

func newMessageTracer(tracer trace.Tracer, propagator propagation.TextMapPropagator, recorder tracing.MessageRecorder, headerAttributes map[string]string) tracing.MessageTracer {
	return newRequestTracer(
		tracer,
		propagator,
		recorder,
		headerAttributes,
		func(span trace.Span, recorder tracing.MessageRecorder) tracing.MessageSpan {
			return &messageSpan{span: span, recorder: recorder}
		},
	)
}
