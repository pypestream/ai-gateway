// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package tracing

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/propagators/autoprop"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"k8s.io/utils/ptr"

	"github.com/envoyproxy/ai-gateway/internal/apischema/cohere"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/json"
	tracing "github.com/envoyproxy/ai-gateway/internal/tracing/api"
)

var (
	startOpts = []oteltrace.SpanStartOption{oteltrace.WithSpanKind(oteltrace.SpanKindServer)}

	req = &openai.ChatCompletionRequest{
		Model: openai.ModelGPT5Nano,
		Messages: []openai.ChatCompletionMessageParamUnion{{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.StringOrUserRoleContentUnion{Value: "Hello!"},
				Role:    openai.ChatMessageRoleUser,
			},
		}},
	}
)

type tracerConstructor[ReqT any, RespT, RespChunkT any] func(oteltrace.Tracer, propagation.TextMapPropagator, map[string]string) tracing.RequestTracer[ReqT, RespT, RespChunkT]

var chatCompletionTracerCtor = func(tr oteltrace.Tracer, prop propagation.TextMapPropagator, headerAttrs map[string]string) tracing.ChatCompletionTracer {
	return newChatCompletionTracer(tr, prop, testChatCompletionRecorder{}, headerAttrs)
}

type requestTracerLifecycleTest[ReqT any, RespT, RespChunkT any] struct {
	constructor      tracerConstructor[ReqT, RespT, RespChunkT]
	req              *ReqT
	headers          map[string]string
	headerAttrs      map[string]string
	reqBody          []byte
	expectedSpanName string
	expectedAttrs    []attribute.KeyValue
	expectedTraceID  string
	expectedSpanType any
	recordAndEnd     func(span tracing.Span[RespT, RespChunkT])
	assertAttrs      func(*testing.T, []attribute.KeyValue)
}

func runRequestTracerLifecycleTest[ReqT any, RespT, RespChunkT any](t *testing.T, tc requestTracerLifecycleTest[ReqT, RespT, RespChunkT]) {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	tp := trace.NewTracerProvider(trace.WithSyncer(exporter))
	tracer := tc.constructor(tp.Tracer("test"), autoprop.NewTextMapPropagator(), tc.headerAttrs)

	carrier := propagation.MapCarrier{}
	span := tracer.StartSpanAndInjectHeaders(t.Context(), tc.headers, carrier, tc.req, tc.reqBody)
	require.IsType(t, tc.expectedSpanType, span)

	tc.recordAndEnd(span)

	spans := exporter.GetSpans()
	require.Len(t, spans, 1)
	actualSpan := spans[0]
	require.Equal(t, tc.expectedSpanName, actualSpan.Name)
	if tc.assertAttrs != nil {
		tc.assertAttrs(t, actualSpan.Attributes)
	} else {
		require.Equal(t, tc.expectedAttrs, actualSpan.Attributes)
	}
	require.Empty(t, actualSpan.Events)

	traceID := actualSpan.SpanContext.TraceID().String()
	if tc.expectedTraceID != "" {
		require.Equal(t, tc.expectedTraceID, traceID)
	}
	spanID := actualSpan.SpanContext.SpanID().String()
	require.Equal(t,
		propagation.MapCarrier{
			"traceparent": fmt.Sprintf("00-%s-%s-01", traceID, spanID),
		}, carrier)
}

func testNoopTracer[ReqT any, RespT, RespChunkT any](t *testing.T, name string, ctor tracerConstructor[ReqT, RespT, RespChunkT], newReq func() *ReqT) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		noopTracer := noop.Tracer{}
		tracer := ctor(noopTracer, autoprop.NewTextMapPropagator(), nil)
		require.IsType(t, tracing.NoopTracer[ReqT, RespT, RespChunkT]{}, tracer)

		headers := map[string]string{}
		carrier := propagation.MapCarrier{}
		req := newReq()
		span := tracer.StartSpanAndInjectHeaders(context.Background(), headers, carrier, req, []byte("{}"))
		require.Nil(t, span)
		require.Empty(t, carrier)
	})
}

func testUnsampledTracer[ReqT any, RespT, RespChunkT any](t *testing.T, name string, ctor tracerConstructor[ReqT, RespT, RespChunkT], newReq func() *ReqT) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		tp := trace.NewTracerProvider(trace.WithSampler(trace.NeverSample()))
		tracer := ctor(tp.Tracer("test"), autoprop.NewTextMapPropagator(), nil)

		headers := map[string]string{}
		carrier := propagation.MapCarrier{}
		req := newReq()
		span := tracer.StartSpanAndInjectHeaders(context.Background(), headers, carrier, req, []byte("{}"))
		require.Nil(t, span)
		require.NotEmpty(t, carrier)
	})
}

func TestChatCompletionTracer_StartSpanAndInjectHeaders(t *testing.T) {
	respBody := &openai.ChatCompletionResponse{
		ID:     "chatcmpl-abc123",
		Object: "chat.completion",
		Model:  "gpt-4.1-nano",
		Choices: []openai.ChatCompletionResponseChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionResponseChoiceMessage{
					Role:    "assistant",
					Content: ptr.To("hello world"),
				},
				FinishReason: "stop",
			},
		},
	}
	respBodyBytes, err := json.Marshal(respBody)
	require.NoError(t, err)
	bodyLen := len(respBodyBytes)

	reqStream := *req
	reqStream.Stream = true

	tests := []struct {
		name             string
		req              *openai.ChatCompletionRequest
		existingHeaders  map[string]string
		expectedSpanName string
		expectedAttrs    []attribute.KeyValue
		expectedTraceID  string
	}{
		{
			name:             "non-streaming request",
			req:              req,
			existingHeaders:  map[string]string{},
			expectedSpanName: "non-stream len: 70",
			expectedAttrs: []attribute.KeyValue{
				attribute.String("req", "stream: false"),
				attribute.Int("reqBodyLen", 70),
				attribute.Int("statusCode", 200),
				attribute.Int("respBodyLen", bodyLen),
			},
		},
		{
			name:             "streaming request",
			req:              &reqStream,
			existingHeaders:  map[string]string{},
			expectedSpanName: "stream len: 84",
			expectedAttrs: []attribute.KeyValue{
				attribute.String("req", "stream: true"),
				attribute.Int("reqBodyLen", 84),
				attribute.Int("statusCode", 200),
				attribute.Int("respBodyLen", bodyLen),
			},
		},
		{
			name: "with existing trace context",
			req:  req,
			existingHeaders: map[string]string{
				"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
			},
			expectedSpanName: "non-stream len: 70",
			expectedAttrs: []attribute.KeyValue{
				attribute.String("req", "stream: false"),
				attribute.Int("reqBodyLen", 70),
				attribute.Int("statusCode", 200),
				attribute.Int("respBodyLen", bodyLen),
			},
			expectedTraceID: "4bf92f3577b34da6a3ce929d0e0e4736",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqBody, err := json.Marshal(tt.req)
			require.NoError(t, err)

			headers := make(map[string]string, len(tt.existingHeaders))
			for k, v := range tt.existingHeaders {
				headers[k] = v
			}

			runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
				constructor:      chatCompletionTracerCtor,
				req:              tt.req,
				headers:          headers,
				reqBody:          reqBody,
				expectedSpanName: tt.expectedSpanName,
				expectedAttrs:    tt.expectedAttrs,
				expectedTraceID:  tt.expectedTraceID,
				expectedSpanType: (*chatCompletionSpan)(nil),
				recordAndEnd: func(span tracing.ChatCompletionSpan) {
					span.RecordResponse(respBody)
					span.EndSpan()
				},
			})
		})
	}
}

func TestRequestTracers_Noop(t *testing.T) {
	testNoopTracer(t, "chat completion", chatCompletionTracerCtor, func() *openai.ChatCompletionRequest {
		return &openai.ChatCompletionRequest{Model: "test"}
	})
}

func TestRequestTracers_Unsampled(t *testing.T) {
	testUnsampledTracer(t, "chat completion", chatCompletionTracerCtor, func() *openai.ChatCompletionRequest {
		return &openai.ChatCompletionRequest{Model: "test"}
	})
}

func TestRequestTracer_HeaderAttributeMapping(t *testing.T) {
	t.Run("chat completion", func(t *testing.T) {
		headers := map[string]string{
			"x-session-id": "abc123",
			"x-user-id":    "user456",
			"x-other":      "ignored",
		}
		reqBody, err := json.Marshal(req)
		require.NoError(t, err)

		spanName := fmt.Sprintf("non-stream len: %d", len(reqBody))

		runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
			constructor:      chatCompletionTracerCtor,
			req:              req,
			headers:          headers,
			headerAttrs:      map[string]string{"x-session-id": "session.id", "x-user-id": "user.id"},
			reqBody:          reqBody,
			expectedSpanName: spanName,
			expectedSpanType: (*chatCompletionSpan)(nil),
			recordAndEnd: func(span tracing.ChatCompletionSpan) {
				span.EndSpan()
			},
			assertAttrs: func(t *testing.T, attrs []attribute.KeyValue) {
				require.Len(t, attrs, 4)
				attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
				for _, attr := range attrs {
					attrMap[attr.Key] = attr.Value
				}
				require.Equal(t, "stream: false", attrMap["req"].AsString())
				require.Equal(t, len(reqBody), int(attrMap["reqBodyLen"].AsInt64()))
				require.Equal(t, "abc123", attrMap["session.id"].AsString())
				require.Equal(t, "user456", attrMap["user.id"].AsString())
			},
		})
	})
}

func TestRequestTracer_SpanNameHeader_Default(t *testing.T) {
	reqBody, err := json.Marshal(req)
	require.NoError(t, err)

	headers := map[string]string{
		defaultSpanNameHeaderName: "custom-span",
	}

	runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
		constructor:      chatCompletionTracerCtor,
		req:              req,
		headers:          headers,
		reqBody:          reqBody,
		expectedSpanName: "custom-span",
		expectedSpanType: (*chatCompletionSpan)(nil),
		recordAndEnd: func(span tracing.ChatCompletionSpan) {
			span.EndSpan()
		},
		assertAttrs: func(t *testing.T, attrs []attribute.KeyValue) {
			attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
			for _, attr := range attrs {
				attrMap[attr.Key] = attr.Value
			}
			require.Equal(t, "stream: false", attrMap["req"].AsString())
			require.Equal(t, len(reqBody), int(attrMap["reqBodyLen"].AsInt64()))
		},
	})
}

func TestRequestTracer_SpanNameHeader_CustomEnv(t *testing.T) {
	t.Setenv("AIGW_SPAN_NAME_HEADER_NAME", "X-Custom-Span-Name")
	prevHeader := spanNameHeaderName
	spanNameHeaderName = resolveSpanNameHeaderName()
	t.Cleanup(func() { spanNameHeaderName = prevHeader })

	reqBody, err := json.Marshal(req)
	require.NoError(t, err)

	headers := map[string]string{
		spanNameHeaderName: "env-configured-span",
	}

	runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
		constructor:      chatCompletionTracerCtor,
		req:              req,
		headers:          headers,
		reqBody:          reqBody,
		expectedSpanName: "env-configured-span",
		expectedSpanType: (*chatCompletionSpan)(nil),
		recordAndEnd: func(span tracing.ChatCompletionSpan) {
			span.EndSpan()
		},
		assertAttrs: func(t *testing.T, attrs []attribute.KeyValue) {
			attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
			for _, attr := range attrs {
				attrMap[attr.Key] = attr.Value
			}
			require.Equal(t, "stream: false", attrMap["req"].AsString())
			require.Equal(t, len(reqBody), int(attrMap["reqBodyLen"].AsInt64()))
		},
	})
}

func TestRequestTracer_MetadataHeader_DefaultPrefix(t *testing.T) {
	t.Setenv("AIGW_METADATA_ATTR_PREFIX", "")
	prevPrefix := metadataAttrPrefix
	metadataAttrPrefix = resolveMetadataAttrPrefix()
	t.Cleanup(func() { metadataAttrPrefix = prevPrefix })

	reqBody, err := json.Marshal(req)
	require.NoError(t, err)
	spanName := fmt.Sprintf("non-stream len: %d", len(reqBody))

	headers := map[string]string{
		"x-ai-metadata": `{"user":{"id":"123"},"tenant":"abc"}`,
	}

	runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
		constructor:      chatCompletionTracerCtor,
		req:              req,
		headers:          headers,
		reqBody:          reqBody,
		expectedSpanName: spanName,
		expectedSpanType: (*chatCompletionSpan)(nil),
		recordAndEnd: func(span tracing.ChatCompletionSpan) {
			span.EndSpan()
		},
		assertAttrs: func(t *testing.T, attrs []attribute.KeyValue) {
			attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
			for _, attr := range attrs {
				attrMap[attr.Key] = attr.Value
			}
			require.Equal(t, "stream: false", attrMap["req"].AsString())
			require.Equal(t, len(reqBody), int(attrMap["reqBodyLen"].AsInt64()))
			require.Equal(t, "123", attrMap["metadata.user.id"].AsString())
			require.Equal(t, "abc", attrMap["metadata.tenant"].AsString())
		},
	})
}

func TestRequestTracer_MetadataHeader_CustomPrefix(t *testing.T) {
	t.Setenv("AIGW_METADATA_ATTR_PREFIX", "custom-prefix")
	prevPrefix := metadataAttrPrefix
	metadataAttrPrefix = resolveMetadataAttrPrefix()
	t.Cleanup(func() { metadataAttrPrefix = prevPrefix })

	reqBody, err := json.Marshal(req)
	require.NoError(t, err)
	spanName := fmt.Sprintf("non-stream len: %d", len(reqBody))

	headers := map[string]string{
		"x-ai-metadata": `{"user":{"id":"123"},"tenant":"abc"}`,
	}

	runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
		constructor:      chatCompletionTracerCtor,
		req:              req,
		headers:          headers,
		reqBody:          reqBody,
		expectedSpanName: spanName,
		expectedSpanType: (*chatCompletionSpan)(nil),
		recordAndEnd: func(span tracing.ChatCompletionSpan) {
			span.EndSpan()
		},
		assertAttrs: func(t *testing.T, attrs []attribute.KeyValue) {
			attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
			for _, attr := range attrs {
				attrMap[attr.Key] = attr.Value
			}
			require.Equal(t, "stream: false", attrMap["req"].AsString())
			require.Equal(t, len(reqBody), int(attrMap["reqBodyLen"].AsInt64()))
			require.Equal(t, "123", attrMap["custom-prefix.user.id"].AsString())
			require.Equal(t, "abc", attrMap["custom-prefix.tenant"].AsString())
		},
	})
}

func TestNewCompletionTracer_BuildsGenericRequestTracer(t *testing.T) {
	tp := trace.NewTracerProvider()
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })

	headerAttrs := map[string]string{"x-session-id": "session.id"}

	tracer := newCompletionTracer(tp.Tracer("test"), autoprop.NewTextMapPropagator(), testCompletionRecorder{}, headerAttrs)
	impl, ok := tracer.(*requestTracerImpl[
		openai.CompletionRequest,
		openai.CompletionResponse,
		openai.CompletionResponse,
	])
	require.True(t, ok)
	require.Equal(t, headerAttrs, impl.headerAttributes)
	require.NotNil(t, impl.newSpan)
	s := tracer.StartSpanAndInjectHeaders(context.Background(), nil, propagation.MapCarrier{}, &openai.CompletionRequest{}, []byte("{}"))
	require.IsType(t, (*completionSpan)(nil), s)
}

func TestNewEmbeddingsTracer_BuildsGenericRequestTracer(t *testing.T) {
	tp := trace.NewTracerProvider()
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })

	headerAttrs := map[string]string{"x-session-id": "session.id"}

	tracer := newEmbeddingsTracer(tp.Tracer("test"), autoprop.NewTextMapPropagator(), testEmbeddingsRecorder{}, headerAttrs)
	impl, ok := tracer.(*requestTracerImpl[
		openai.EmbeddingRequest,
		openai.EmbeddingResponse,
		struct{},
	])
	require.True(t, ok)
	require.Equal(t, headerAttrs, impl.headerAttributes)
	require.NotNil(t, impl.newSpan)
}

func TestNewRerankTracer_BuildsGenericRequestTracer(t *testing.T) {
	tp := trace.NewTracerProvider()
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })

	headerAttrs := map[string]string{"x-session-id": "session.id"}

	tracer := newRerankTracer(tp.Tracer("test"), autoprop.NewTextMapPropagator(), testRerankTracerRecorder{}, headerAttrs)
	impl, ok := tracer.(*requestTracerImpl[
		cohere.RerankV2Request,
		cohere.RerankV2Response,
		struct{},
	])
	require.True(t, ok)
	require.Equal(t, headerAttrs, impl.headerAttributes)
	require.NotNil(t, impl.newSpan)
	s := tracer.StartSpanAndInjectHeaders(context.Background(), nil, propagation.MapCarrier{}, &cohere.RerankV2Request{
		TopN: ptr.To(1),
	}, []byte("{}"))
	require.IsType(t, (*rerankSpan)(nil), s)
}

func TestNewImageGenerationTracer_BuildsGenericRequestTracer(t *testing.T) {
	tp := trace.NewTracerProvider()
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })

	tracer := newImageGenerationTracer(tp.Tracer("test"), autoprop.NewTextMapPropagator(), testImageGenerationRecorder{})
	impl, ok := tracer.(*requestTracerImpl[
		openai.ImageGenerationRequest,
		openai.ImageGenerationResponse,
		struct{},
	])
	require.True(t, ok)
	require.Nil(t, impl.headerAttributes)
	require.NotNil(t, impl.newSpan)
	s := tracer.StartSpanAndInjectHeaders(context.Background(), nil, propagation.MapCarrier{}, &openai.ImageGenerationRequest{}, []byte("{}"))
	require.IsType(t, (*imageGenerationSpan)(nil), s)
}

func TestResponsesTracer_BuildsGenericRequestTracer(t *testing.T) {
	tp := trace.NewTracerProvider()
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })

	headerAttrs := map[string]string{"x-session-id": "session.id"}

	tracer := newResponsesTracer(tp.Tracer("test"), autoprop.NewTextMapPropagator(), testResponsesRecorder{}, headerAttrs)
	impl, ok := tracer.(*requestTracerImpl[
		openai.ResponseRequest,
		openai.Response,
		openai.ResponseStreamEventUnion,
	])
	require.True(t, ok)
	require.Equal(t, headerAttrs, impl.headerAttributes)
	require.NotNil(t, impl.newSpan)
	s := tracer.StartSpanAndInjectHeaders(context.Background(), nil, propagation.MapCarrier{}, &openai.ResponseRequest{}, []byte("{}"))
	require.IsType(t, (*responsesSpan)(nil), s)
}

type testChatCompletionRecorder struct{}

func (r testChatCompletionRecorder) RecordResponseChunks(span oteltrace.Span, chunks []*openai.ChatCompletionResponseChunk) {
	span.SetAttributes(attribute.Int("eventCount", len(chunks)))
}

func (r testChatCompletionRecorder) RecordResponseOnError(span oteltrace.Span, statusCode int, body []byte) {
	span.SetAttributes(attribute.Int("statusCode", statusCode))
	span.SetAttributes(attribute.String("errorBody", string(body)))
}

func (testChatCompletionRecorder) StartParams(req *openai.ChatCompletionRequest, body []byte) (spanName string, opts []oteltrace.SpanStartOption) {
	if req.Stream {
		return fmt.Sprintf("stream len: %d", len(body)), startOpts
	}
	return fmt.Sprintf("non-stream len: %d", len(body)), startOpts
}

func (testChatCompletionRecorder) RecordRequest(span oteltrace.Span, req *openai.ChatCompletionRequest, body []byte) {
	span.SetAttributes(attribute.String("req", fmt.Sprintf("stream: %v", req.Stream)))
	span.SetAttributes(attribute.Int("reqBodyLen", len(body)))
}

func (testChatCompletionRecorder) RecordResponse(span oteltrace.Span, resp *openai.ChatCompletionResponse) {
	span.SetAttributes(attribute.Int("statusCode", 200))
	body, err := json.Marshal(resp)
	if err != nil {
		panic(err)
	}
	span.SetAttributes(attribute.Int("respBodyLen", len(body)))
}

var _ tracing.EmbeddingsRecorder = testEmbeddingsRecorder{}

type testEmbeddingsRecorder struct {
	tracing.NoopChunkRecorder[struct{}]
}

func (testEmbeddingsRecorder) RecordResponseOnError(span oteltrace.Span, statusCode int, body []byte) {
	span.SetAttributes(attribute.Int("statusCode", statusCode))
	span.SetAttributes(attribute.String("errorBody", string(body)))
}

func (testEmbeddingsRecorder) StartParams(_ *openai.EmbeddingRequest, _ []byte) (spanName string, opts []oteltrace.SpanStartOption) {
	return "Embeddings", startOpts
}

func (testEmbeddingsRecorder) RecordRequest(span oteltrace.Span, req *openai.EmbeddingRequest, body []byte) {
	span.SetAttributes(attribute.String("model", req.Model))
	span.SetAttributes(attribute.Int("reqBodyLen", len(body)))
}

func (testEmbeddingsRecorder) RecordResponse(span oteltrace.Span, resp *openai.EmbeddingResponse) {
	span.SetAttributes(attribute.Int("statusCode", 200))
	body, err := json.Marshal(resp)
	if err != nil {
		panic(err)
	}
	span.SetAttributes(attribute.Int("respBodyLen", len(body)))
}

type testCompletionRecorder struct{}

func (r testCompletionRecorder) RecordResponseChunks(span oteltrace.Span, chunks []*openai.CompletionResponse) {
	span.SetAttributes(attribute.Int("eventCount", len(chunks)))
}

func (r testCompletionRecorder) RecordResponseOnError(span oteltrace.Span, statusCode int, body []byte) {
	span.SetAttributes(attribute.Int("statusCode", statusCode))
	span.SetAttributes(attribute.String("errorBody", string(body)))
}

func (testCompletionRecorder) StartParams(req *openai.CompletionRequest, body []byte) (spanName string, opts []oteltrace.SpanStartOption) {
	if req.Stream {
		return fmt.Sprintf("completion-stream len: %d", len(body)), startOpts
	}
	return fmt.Sprintf("completion-non-stream len: %d", len(body)), startOpts
}

func (testCompletionRecorder) RecordRequest(span oteltrace.Span, req *openai.CompletionRequest, body []byte) {
	span.SetAttributes(attribute.String("req", fmt.Sprintf("stream: %v", req.Stream)))
	span.SetAttributes(attribute.Int("reqBodyLen", len(body)))
}

func (testCompletionRecorder) RecordResponse(span oteltrace.Span, resp *openai.CompletionResponse) {
	span.SetAttributes(attribute.Int("statusCode", 200))
	body, err := json.Marshal(resp)
	if err != nil {
		panic(err)
	}
	span.SetAttributes(attribute.Int("respBodyLen", len(body)))
}

// Mock recorder for testing image generation span
type testImageGenerationRecorder struct {
	tracing.NoopChunkRecorder[struct{}]
}

func (r testImageGenerationRecorder) StartParams(_ *openai.ImageGenerationRequest, _ []byte) (string, []oteltrace.SpanStartOption) {
	return "ImagesResponse", nil
}

func (r testImageGenerationRecorder) RecordRequest(span oteltrace.Span, req *openai.ImageGenerationRequest, _ []byte) {
	span.SetAttributes(
		attribute.String("model", req.Model),
		attribute.String("prompt", req.Prompt),
		attribute.String("size", req.Size),
	)
}

func (r testImageGenerationRecorder) RecordResponse(span oteltrace.Span, resp *openai.ImageGenerationResponse) {
	respBytes, _ := json.Marshal(resp)
	span.SetAttributes(
		attribute.Int("statusCode", 200),
		attribute.Int("respBodyLen", len(respBytes)),
	)
}

func (r testImageGenerationRecorder) RecordResponseOnError(span oteltrace.Span, statusCode int, body []byte) {
	span.SetAttributes(
		attribute.Int("statusCode", statusCode),
		attribute.String("errorBody", string(body)),
	)
}

type testRerankTracerRecorder struct {
	tracing.NoopChunkRecorder[struct{}]
}

func (testRerankTracerRecorder) StartParams(*cohere.RerankV2Request, []byte) (string, []oteltrace.SpanStartOption) {
	return "Rerank", []oteltrace.SpanStartOption{oteltrace.WithSpanKind(oteltrace.SpanKindServer)}
}

func (testRerankTracerRecorder) RecordRequest(span oteltrace.Span, req *cohere.RerankV2Request, body []byte) {
	span.SetAttributes(
		attribute.String("model", req.Model),
		attribute.String("query", req.Query),
		attribute.Int("top_n", *req.TopN),
		attribute.Int("reqBodyLen", len(body)),
	)
}

func (testRerankTracerRecorder) RecordResponse(span oteltrace.Span, resp *cohere.RerankV2Response) {
	span.SetAttributes(attribute.Int("statusCode", 200))
	b, _ := json.Marshal(resp)
	span.SetAttributes(attribute.Int("respBodyLen", len(b)))
}

func (testRerankTracerRecorder) RecordResponseOnError(span oteltrace.Span, statusCode int, body []byte) {
	span.SetAttributes(attribute.Int("statusCode", statusCode))
	span.SetAttributes(attribute.String("errorBody", string(body)))
}

// Mock recorder for testing responses span
type testResponsesRecorder struct {
	tracing.NoopChunkRecorder[openai.ResponseStreamEventUnion]
}

func (r testResponsesRecorder) StartParams(_ *openai.ResponseRequest, _ []byte) (string, []oteltrace.SpanStartOption) {
	return "Responses", startOpts
}

func (r testResponsesRecorder) RecordRequest(span oteltrace.Span, req *openai.ResponseRequest, body []byte) {
	span.SetAttributes(
		attribute.String("model", req.Model),
		attribute.Int("reqBodyLen", len(body)),
	)
}

func (r testResponsesRecorder) RecordResponse(span oteltrace.Span, resp *openai.Response) {
	respBytes, err := json.Marshal(resp)
	if err != nil {
		panic(err)
	}
	span.SetAttributes(
		attribute.Int("statusCode", 200),
		attribute.Int("respBodyLen", len(respBytes)),
	)
}

func (r testResponsesRecorder) RecordResponseOnError(span oteltrace.Span, statusCode int, body []byte) {
	span.SetAttributes(
		attribute.Int("statusCode", statusCode),
		attribute.String("errorBody", string(body)),
	)
}

// TestFlattenJSON tests the flattenJSON function with various input types including arrays.
func TestFlattenJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    any
		expected map[string]string
	}{
		{
			name: "simple object",
			input: map[string]any{
				"key1": "value1",
				"key2": "value2",
			},
			expected: map[string]string{
				"key1": "value1",
				"key2": "value2",
			},
		},
		{
			name: "nested object",
			input: map[string]any{
				"user": map[string]any{
					"id":   "123",
					"name": "John",
				},
			},
			expected: map[string]string{
				"user.id":   "123",
				"user.name": "John",
			},
		},
		{
			name: "deeply nested object",
			input: map[string]any{
				"a": map[string]any{
					"b": map[string]any{
						"c": "value",
					},
				},
			},
			expected: map[string]string{
				"a.b.c": "value",
			},
		},
		{
			name: "simple array",
			input: map[string]any{
				"tags": []any{"tag1", "tag2", "tag3"},
			},
			expected: map[string]string{
				"tags.0": "tag1",
				"tags.1": "tag2",
				"tags.2": "tag3",
			},
		},
		{
			name:  "top-level array",
			input: []any{"item1", "item2", "item3"},
			expected: map[string]string{
				"0": "item1",
				"1": "item2",
				"2": "item3",
			},
		},
		{
			name: "array of objects",
			input: map[string]any{
				"users": []any{
					map[string]any{"id": "1", "name": "Alice"},
					map[string]any{"id": "2", "name": "Bob"},
				},
			},
			expected: map[string]string{
				"users.0.id":   "1",
				"users.0.name": "Alice",
				"users.1.id":   "2",
				"users.1.name": "Bob",
			},
		},
		{
			name: "nested array",
			input: map[string]any{
				"matrix": []any{
					[]any{"a", "b"},
					[]any{"c", "d"},
				},
			},
			expected: map[string]string{
				"matrix.0.0": "a",
				"matrix.0.1": "b",
				"matrix.1.0": "c",
				"matrix.1.1": "d",
			},
		},
		{
			name: "mixed object and array",
			input: map[string]any{
				"user": map[string]any{
					"id":   "123",
					"tags": []any{"vip", "beta"},
				},
				"count": float64(42),
			},
			expected: map[string]string{
				"user.id":     "123",
				"user.tags.0": "vip",
				"user.tags.1": "beta",
				"count":       "42",
			},
		},
		{
			name: "various types",
			input: map[string]any{
				"string": "text",
				"number": float64(123.45),
				"bool":   true,
				"null":   nil,
			},
			expected: map[string]string{
				"string": "text",
				"number": "123.45",
				"bool":   "true",
				"null":   "",
			},
		},
		{
			name: "empty array",
			input: map[string]any{
				"empty": []any{},
			},
			expected: map[string]string{},
		},
		{
			name:     "empty object",
			input:    map[string]any{},
			expected: map[string]string{},
		},
		{
			name: "array with mixed types",
			input: map[string]any{
				"mixed": []any{"string", float64(123), true, nil},
			},
			expected: map[string]string{
				"mixed.0": "string",
				"mixed.1": "123",
				"mixed.2": "true",
				"mixed.3": "",
			},
		},
		{
			name: "complex nested structure",
			input: map[string]any{
				"service": map[string]any{
					"name": "api",
					"endpoints": []any{
						map[string]any{
							"path":    "/users",
							"methods": []any{"GET", "POST"},
						},
						map[string]any{
							"path":    "/items",
							"methods": []any{"GET"},
						},
					},
					"active": true,
				},
			},
			expected: map[string]string{
				"service.name":                "api",
				"service.endpoints.0.path":    "/users",
				"service.endpoints.0.methods.0": "GET",
				"service.endpoints.0.methods.1": "POST",
				"service.endpoints.1.path":    "/items",
				"service.endpoints.1.methods.0": "GET",
				"service.active":              "true",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := make(map[string]string)
			flattenJSON("", tt.input, result)
			require.Equal(t, tt.expected, result)
		})
	}
}

// TestParseMetadataHeader tests the parseMetadataHeader function with various JSON inputs.
func TestParseMetadataHeader(t *testing.T) {
	// Save and restore the original prefix
	origPrefix := metadataAttrPrefix
	t.Cleanup(func() { metadataAttrPrefix = origPrefix })
	metadataAttrPrefix = "metadata."

	tests := []struct {
		name     string
		input    string
		expected []attribute.KeyValue
	}{
		{
			name:  "simple object",
			input: `{"key":"value"}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.key", "value"),
			},
		},
		{
			name:  "nested object",
			input: `{"user":{"id":"123","name":"John"}}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.user.id", "123"),
				attribute.String("metadata.user.name", "John"),
			},
		},
		{
			name:  "array in object",
			input: `{"tags":["tag1","tag2","tag3"]}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.tags.0", "tag1"),
				attribute.String("metadata.tags.1", "tag2"),
				attribute.String("metadata.tags.2", "tag3"),
			},
		},
		{
			name:  "array of objects",
			input: `{"users":[{"id":"1"},{"id":"2"}]}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.users.0.id", "1"),
				attribute.String("metadata.users.1.id", "2"),
			},
		},
		{
			name:  "mixed types",
			input: `{"str":"text","num":42,"bool":true,"null":null}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.str", "text"),
				attribute.String("metadata.num", "42"),
				attribute.String("metadata.bool", "true"),
				attribute.String("metadata.null", ""),
			},
		},
		{
			name:  "invalid JSON",
			input: `{invalid json}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.raw", "{invalid json}"),
			},
		},
		{
			name:  "empty object",
			input: `{}`,
			expected: []attribute.KeyValue{},
		},
		{
			name:  "complex nested structure",
			input: `{"service":{"name":"api","endpoints":[{"path":"/users","methods":["GET","POST"]},{"path":"/items"}]}}`,
			expected: []attribute.KeyValue{
				attribute.String("metadata.service.name", "api"),
				attribute.String("metadata.service.endpoints.0.path", "/users"),
				attribute.String("metadata.service.endpoints.0.methods.0", "GET"),
				attribute.String("metadata.service.endpoints.0.methods.1", "POST"),
				attribute.String("metadata.service.endpoints.1.path", "/items"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseMetadataHeader(tt.input)

			// Convert both to maps for easier comparison (order doesn't matter)
			resultMap := make(map[attribute.Key]attribute.Value)
			for _, attr := range result {
				resultMap[attr.Key] = attr.Value
			}

			expectedMap := make(map[attribute.Key]attribute.Value)
			for _, attr := range tt.expected {
				expectedMap[attr.Key] = attr.Value
			}

			require.Equal(t, expectedMap, resultMap)
		})
	}
}

// TestParseMetadataHeader_LargeStructure tests that very large metadata structures
// are stored as JSON to avoid hitting span attribute limits.
func TestParseMetadataHeader_LargeStructure(t *testing.T) {
	// Save and restore the original prefix
	origPrefix := metadataAttrPrefix
	t.Cleanup(func() { metadataAttrPrefix = origPrefix })
	metadataAttrPrefix = "metadata."

	// Create a JSON structure that will exceed maxFlattenedAttributes
	largeArray := make([]map[string]any, 20) // 20 items
	for i := 0; i < 20; i++ {
		largeArray[i] = map[string]any{
			"id":       fmt.Sprintf("item-%d", i),
			"name":     fmt.Sprintf("Name %d", i),
			"value":    i * 100,
			"active":   i%2 == 0,
			"metadata": fmt.Sprintf("meta-%d", i),
		}
	}
	// 20 items * 5 fields = 100 attributes (exceeds maxFlattenedAttributes of 50)

	input := map[string]any{
		"items": largeArray,
	}
	jsonBytes, err := json.Marshal(input)
	require.NoError(t, err)
	jsonString := string(jsonBytes)

	result := parseMetadataHeader(jsonString)

	// Should return JSON fallback instead of flattened attributes
	require.Len(t, result, 2)

	resultMap := make(map[attribute.Key]attribute.Value)
	for _, attr := range result {
		resultMap[attr.Key] = attr.Value
	}

	// Check that it fell back to JSON storage
	require.Contains(t, resultMap, attribute.Key("metadata.json"))
	require.Equal(t, jsonString, resultMap[attribute.Key("metadata.json")].AsString())

	// Check that it includes the count
	require.Contains(t, resultMap, attribute.Key("metadata.flattened_count"))
	require.Equal(t, int64(100), resultMap[attribute.Key("metadata.flattened_count")].AsInt64())
}

// TestParseMetadataHeader_BelowLimit tests that structures below the limit
// are still flattened normally.
func TestParseMetadataHeader_BelowLimit(t *testing.T) {
	// Save and restore the original prefix
	origPrefix := metadataAttrPrefix
	t.Cleanup(func() { metadataAttrPrefix = origPrefix })
	metadataAttrPrefix = "metadata."

	// Create a structure with exactly maxFlattenedAttributes (should still flatten)
	items := make([]map[string]any, 10) // 10 items
	for i := 0; i < 10; i++ {
		items[i] = map[string]any{
			"id":   fmt.Sprintf("item-%d", i),
			"name": fmt.Sprintf("Name %d", i),
			"val":  i * 10,
		}
	}
	// 10 items * 3 fields = 30 attributes (below maxFlattenedAttributes of 50)

	input := map[string]any{
		"items": items,
	}
	jsonBytes, err := json.Marshal(input)
	require.NoError(t, err)

	result := parseMetadataHeader(string(jsonBytes))

	// Should be flattened (not JSON fallback)
	require.Equal(t, 30, len(result))

	// Verify some of the flattened attributes
	resultMap := make(map[attribute.Key]attribute.Value)
	for _, attr := range result {
		resultMap[attr.Key] = attr.Value
	}

	require.Equal(t, "item-0", resultMap[attribute.Key("metadata.items.0.id")].AsString())
	require.Equal(t, "item-9", resultMap[attribute.Key("metadata.items.9.id")].AsString())
	require.Equal(t, "Name 5", resultMap[attribute.Key("metadata.items.5.name")].AsString())

	// Should NOT have the JSON fallback keys
	require.NotContains(t, resultMap, attribute.Key("metadata.json"))
	require.NotContains(t, resultMap, attribute.Key("metadata.flattened_count"))
}

// TestParseMetadataHeader_ExactlyAtLimit tests the boundary condition.
func TestParseMetadataHeader_ExactlyAtLimit(t *testing.T) {
	// Save and restore the original prefix
	origPrefix := metadataAttrPrefix
	t.Cleanup(func() { metadataAttrPrefix = origPrefix })
	metadataAttrPrefix = "metadata."

	// Create exactly maxFlattenedAttributes (50) attributes
	items := make(map[string]any)
	for i := 0; i < 50; i++ {
		items[fmt.Sprintf("field_%d", i)] = fmt.Sprintf("value_%d", i)
	}

	input := map[string]any{
		"data": items,
	}
	jsonBytes, err := json.Marshal(input)
	require.NoError(t, err)

	result := parseMetadataHeader(string(jsonBytes))

	// Should be flattened (50 is at the limit, not over)
	require.Equal(t, 50, len(result))

	// Verify it's flattened, not JSON
	resultMap := make(map[attribute.Key]attribute.Value)
	for _, attr := range result {
		resultMap[attr.Key] = attr.Value
	}
	require.NotContains(t, resultMap, attribute.Key("metadata.json"))
}

// TestParseMetadataHeader_OneOverLimit tests exceeding the limit by one.
func TestParseMetadataHeader_OneOverLimit(t *testing.T) {
	// Save and restore the original prefix
	origPrefix := metadataAttrPrefix
	t.Cleanup(func() { metadataAttrPrefix = origPrefix })
	metadataAttrPrefix = "metadata."

	// Create maxFlattenedAttributes + 1 (51) attributes
	items := make(map[string]any)
	for i := 0; i < 51; i++ {
		items[fmt.Sprintf("field_%d", i)] = fmt.Sprintf("value_%d", i)
	}

	input := map[string]any{
		"data": items,
	}
	jsonBytes, err := json.Marshal(input)
	require.NoError(t, err)
	jsonString := string(jsonBytes)

	result := parseMetadataHeader(jsonString)

	// Should fall back to JSON (51 exceeds the limit of 50)
	require.Len(t, result, 2)

	resultMap := make(map[attribute.Key]attribute.Value)
	for _, attr := range result {
		resultMap[attr.Key] = attr.Value
	}

	require.Contains(t, resultMap, attribute.Key("metadata.json"))
	require.Equal(t, jsonString, resultMap[attribute.Key("metadata.json")].AsString())
	require.Equal(t, int64(51), resultMap[attribute.Key("metadata.flattened_count")].AsInt64())
}

// TestRequestTracer_MetadataHeaderWithArrays tests that array metadata is properly flattened
// and added as span attributes.
func TestRequestTracer_MetadataHeaderWithArrays(t *testing.T) {
	reqBody, err := json.Marshal(req)
	require.NoError(t, err)
	spanName := fmt.Sprintf("non-stream len: %d", len(reqBody))

	tests := []struct {
		name            string
		metadataJSON    string
		expectedAttrs   map[string]string
	}{
		{
			name:         "simple array",
			metadataJSON: `{"tags":["tag1","tag2","tag3"]}`,
			expectedAttrs: map[string]string{
				"metadata.tags.0": "tag1",
				"metadata.tags.1": "tag2",
				"metadata.tags.2": "tag3",
			},
		},
		{
			name:         "array of objects",
			metadataJSON: `{"items":[{"id":"1","type":"A"},{"id":"2","type":"B"}]}`,
			expectedAttrs: map[string]string{
				"metadata.items.0.id":   "1",
				"metadata.items.0.type": "A",
				"metadata.items.1.id":   "2",
				"metadata.items.1.type": "B",
			},
		},
		{
			name:         "nested arrays",
			metadataJSON: `{"matrix":[["a","b"],["c","d"]]}`,
			expectedAttrs: map[string]string{
				"metadata.matrix.0.0": "a",
				"metadata.matrix.0.1": "b",
				"metadata.matrix.1.0": "c",
				"metadata.matrix.1.1": "d",
			},
		},
		{
			name:         "mixed object and array",
			metadataJSON: `{"user":{"id":"123","roles":["admin","user"]},"count":5}`,
			expectedAttrs: map[string]string{
				"metadata.user.id":      "123",
				"metadata.user.roles.0": "admin",
				"metadata.user.roles.1": "user",
				"metadata.count":        "5",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			headers := map[string]string{
				"x-ai-metadata": tt.metadataJSON,
			}

			runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
				constructor:      chatCompletionTracerCtor,
				req:              req,
				headers:          headers,
				reqBody:          reqBody,
				expectedSpanName: spanName,
				expectedSpanType: (*chatCompletionSpan)(nil),
				recordAndEnd: func(span tracing.ChatCompletionSpan) {
					span.EndSpan()
				},
				assertAttrs: func(t *testing.T, attrs []attribute.KeyValue) {
					attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
					for _, attr := range attrs {
						attrMap[attr.Key] = attr.Value
					}

					// Check metadata attributes
					for key, expectedValue := range tt.expectedAttrs {
						actualValue, ok := attrMap[attribute.Key(key)]
						require.True(t, ok, "Expected attribute %s not found", key)
						require.Equal(t, expectedValue, actualValue.AsString(), "Mismatch for attribute %s", key)
					}
				},
			})
		})
	}
}

// TestRequestTracer_MetadataHeader_LargeStructure tests that very large metadata
// is handled gracefully without hitting attribute limits.
func TestRequestTracer_MetadataHeader_LargeStructure(t *testing.T) {
	reqBody, err := json.Marshal(req)
	require.NoError(t, err)
	spanName := fmt.Sprintf("non-stream len: %d", len(reqBody))

	// Create a large structure that exceeds maxFlattenedAttributes
	largeArray := make([]map[string]any, 20)
	for i := 0; i < 20; i++ {
		largeArray[i] = map[string]any{
			"id":       fmt.Sprintf("item-%d", i),
			"name":     fmt.Sprintf("Name %d", i),
			"value":    i * 100,
			"active":   i%2 == 0,
			"metadata": fmt.Sprintf("meta-%d", i),
		}
	}
	largeMetadata := map[string]any{
		"items": largeArray,
	}
	largeJSON, err := json.Marshal(largeMetadata)
	require.NoError(t, err)

	headers := map[string]string{
		"x-ai-metadata": string(largeJSON),
	}

	runRequestTracerLifecycleTest(t, requestTracerLifecycleTest[openai.ChatCompletionRequest, openai.ChatCompletionResponse, openai.ChatCompletionResponseChunk]{
		constructor:      chatCompletionTracerCtor,
		req:              req,
		headers:          headers,
		reqBody:          reqBody,
		expectedSpanName: spanName,
		expectedSpanType: (*chatCompletionSpan)(nil),
		recordAndEnd: func(span tracing.ChatCompletionSpan) {
			span.EndSpan()
		},
		assertAttrs: func(t *testing.T, attrs []attribute.KeyValue) {
			attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
			for _, attr := range attrs {
				attrMap[attr.Key] = attr.Value
			}

			// Should have fallen back to JSON storage
			require.Contains(t, attrMap, attribute.Key("metadata.json"))
			require.Equal(t, string(largeJSON), attrMap[attribute.Key("metadata.json")].AsString())

			// Should include the count of what would have been created
			require.Contains(t, attrMap, attribute.Key("metadata.flattened_count"))
			require.Equal(t, int64(100), attrMap[attribute.Key("metadata.flattened_count")].AsInt64())

			// Should NOT have individual flattened attributes
			require.NotContains(t, attrMap, attribute.Key("metadata.items.0.id"))
			require.NotContains(t, attrMap, attribute.Key("metadata.items.0.name"))
		},
	})
}

// TestParseMetadataHeader_CustomLimit tests that the limit can be configured via env var.
func TestParseMetadataHeader_CustomLimit(t *testing.T) {
	// Save and restore
	origPrefix := metadataAttrPrefix
	origLimit := maxFlattenedAttributes
	t.Cleanup(func() {
		metadataAttrPrefix = origPrefix
		maxFlattenedAttributes = origLimit
	})
	metadataAttrPrefix = "metadata."

	// Set custom limit to 10
	t.Setenv("AIGW_MAX_FLATTENED_ATTRIBUTES", "10")
	maxFlattenedAttributes = resolveMaxFlattenedAttributes()

	// Create 15 attributes (exceeds custom limit of 10)
	items := make(map[string]any)
	for i := 0; i < 15; i++ {
		items[fmt.Sprintf("field_%d", i)] = fmt.Sprintf("value_%d", i)
	}
	input := map[string]any{"data": items}
	jsonBytes, err := json.Marshal(input)
	require.NoError(t, err)
	jsonString := string(jsonBytes)

	result := parseMetadataHeader(jsonString)

	// Should fall back to JSON (15 exceeds custom limit of 10)
	require.Len(t, result, 2)
	resultMap := make(map[attribute.Key]attribute.Value)
	for _, attr := range result {
		resultMap[attr.Key] = attr.Value
	}
	require.Contains(t, resultMap, attribute.Key("metadata.json"))
	require.Equal(t, int64(15), resultMap[attribute.Key("metadata.flattened_count")].AsInt64())
}
