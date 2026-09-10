# Gateway-Managed Prompt Cache Lifecycle (`cacheConfig`)

Status: **Design / investigation. Not accepted, not implemented.**
Author: Pypestream platform
Date: 2026-09-10
Fork: `pypestream/ai-gateway` (branch `design/gateway-managed-prompt-cache`)
Relates to: [`008-gemini-context-caching`](../008-gemini-context-caching/proposal.md) (accepted upstream, docs only, unimplemented)

## Table of Contents

<!-- toc -->

- [Summary](#summary)
- [Headline recommendation](#headline-recommendation)
- [The requirement as stated](#the-requirement-as-stated)
- [Measurements](#measurements)
- [Where this can physically live in the codebase](#where-this-can-physically-live-in-the-codebase)
- [Interface: critique and recommendation](#interface-critique-and-recommendation)
- [Reconciliation with proposal 008](#reconciliation-with-proposal-008)
- [Per-provider mapping](#per-provider-mapping)
- [Cache identity and reuse](#cache-identity-and-reuse)
- [State](#state)
- [Concurrency](#concurrency)
- [Lifecycle](#lifecycle)
- [Latency and the two timeout ceilings](#latency-and-the-two-timeout-ceilings)
- [Fail-open vs fail-closed](#fail-open-vs-fail-closed)
- [Region and project affinity](#region-and-project-affinity)
- [Observability plan](#observability-plan)
- [Cost model](#cost-model)
- [File-by-file change inventory](#file-by-file-change-inventory)
- [Upstream-ability](#upstream-ability)
- [Risks](#risks)
- [Recommendation](#recommendation)
- [What could not be verified](#what-could-not-be-verified)

<!-- /toc -->

## Summary

The ask: a caller sends `cacheConfig: {use: true, prompt: "..."}` on an OpenAI-shaped
`/v1/chat/completions` request, and the gateway takes over the provider's explicit
prompt-cache lifecycle — create, reuse, reference, expire — so the caller never sees a
provider cache id.

This document is the design and the honest list of hard parts. It reaches a negative
recommendation on the headline feature and a positive one on a much smaller subset.

Three findings drive everything below.

1. **"Explicit prompt-cache lifecycle" is a Gemini-only feature wearing a cross-provider
   costume.** Of the four provider families the gateway supports, exactly one (GCP Vertex
   AI / Gemini) has a cache *resource* with a lifecycle to own. Anthropic and Bedrock use
   inline breakpoints — there is nothing to create, name, or delete. OpenAI has no explicit
   mechanism at all. A single `cacheConfig` field cannot mean the same thing in all four.

2. **The Vertex `cachedContents` create API is, right now, too unreliable to sit in a
   synchronous request path.** Measured over 25 sequential creates of identical, valid,
   above-minimum content in `us-central1` on `gemini-2.5-flash`: **17 of 25 failed (68%)**
   with a spurious `400 "The cached content is of 1 tokens. The minimum token count to
   start explicit caching is 1024."` The content was ~2,442 tokens. This is not a payload
   bug — the same payload succeeds intermittently, and succeeded 6/6 in `europe-west4`.

3. **Implicit caching already delivers the same discount, for free, and already works
   through the gateway.** Explicit caching's *only* marginal advantages are guaranteed hits
   and not resending the context. It is strictly more expensive than implicit caching
   whenever implicit caching hits, because explicit adds an hourly storage meter that
   implicit does not have.

## Headline recommendation

**Do not build gateway-owned explicit cache lifecycle now.** Build the cheap subset that
carries most of the value and none of the risk:

| # | Deliverable | Size | Why |
|---|---|---|---|
| 1 | Cache observability (response header, metrics, span attrs) | S | Fixes the actual reported pain — silent drops. No provider dependency. |
| 2 | `cacheConfig` desugars to `cache_control` for the Anthropic family | S | Works today, costs nothing, gives callers the requested field shape. |
| 3 | `cachedContent` pass-through on `GCPVertexAIVendorFields` | XS (~30 lines) | Unblocks callers who manage their own Vertex caches. Natural upstream contribution. |
| 4 | Gateway-owned Vertex cache lifecycle | L | **Defer.** Blocked on Vertex reliability and on a demonstrated implicit-cache miss. |

Rationale is developed in [Cost model](#cost-model) and [Recommendation](#recommendation).

## The requirement as stated

```json
{
  "model": "...",
  "messages": [...],
  "cacheConfig": { "use": true, "prompt": "<long stable prompt text>" }
}
```

Current behaviour of this exact request: `openai.ChatCompletionRequest` does not set
`DisallowUnknownFields`, so `cacheConfig` is **silently discarded at unmarshal**. The
caller gets `HTTP 200`, full-price tokens, and no warning. Per
`site/docs/capabilities/llm-integrations/vendor-specific-fields.md`, ignoring unsupported
fields is intended behaviour, so this is a feature gap rather than a bug — but the silence
is the thing that has repeatedly burned people here, and any design must fix it first.

## Measurements

All measured 2026-09-10 against Vertex project `admin-295215`, service account
`vertex-ai@admin-295215.iam.gserviceaccount.com`, model `gemini-2.5-flash`, via the raw
REST API (bypassing the gateway). Scripts in the session scratchpad; the pattern follows
`/Users/shababqaisar/pypestream/gemini_explicit_cache.py`.

### Cache-create reliability

25 sequential creates, identical valid payload (~2,442 tokens, above the 1,024 minimum),
`us-central1`:

```
us-central1 gemini-2.5-flash: 8/25 succeeded (32%), failure rate 68%
  success latency  min=2.07s med=2.14s max=2.79s
  failure latency  min=2.32s med=2.50s max=4.41s
```

Every failure was the same spurious `400 INVALID_ARGUMENT`:

```
The cached content is of 1 tokens. The minimum token count to start explicit caching is 1024.
```

Cross-region, 6 creates each (two runs, showing the noise):

| Region | Run 1 | Run 2 |
|---|---|---|
| `us-central1` | 2/6 | 2/6 |
| `us-east1` | 1/6 | 4/6 |
| `europe-west4` | 6/6 | 6/6 |

Ruling out the obvious explanations:

- **Not rate-related.** 8 creates spaced 5s apart still failed 2/8.
- **Not size-related.** 14,080-char payload failed in one run and succeeded in another;
  a 140,006-char payload succeeded. Chunking into multiple `parts` changed nothing.
- **Not content-repetition-related.** Both repetitive and lexically varied corpora
  reproduce both outcomes.
- **Retry is a weak mitigation.** One immediate retry recovered only **1 of 5** failures,
  at ~2.5s per attempt.

### Vertex does not deduplicate

12 creates of byte-identical content produced **12 distinct cache resources**, each
independently billed for storage. Any deduplication must be implemented by the gateway.

### Stale cache references hard-fail the completion

This is the most dangerous behaviour for a stateful design. Referencing a cache that has
been deleted (which is what TTL expiry looks like from outside) does **not** degrade to an
uncached call:

```
live cache            -> 200  prompt=2447 cached=2442
delete                -> 200
deleted/expired cache -> 400  INVALID_ARGUMENT "Invalid resource state for cache content <id>."
bogus cache name      -> 404  NOT_FOUND "Not found: cached content metadata for <id>."
```

So a single stale entry in the gateway's mapping turns into a user-visible `400` on the
completion. Two distinct error signatures must both be caught, evicted, and retried
without the cache reference.

### Lookup-path latencies

| Operation | Latency |
|---|---|
| `POST cachedContents` (create) | ~2.1–2.8s typical, 9.4s observed worst case |
| `GET cachedContents/{id}` | ~2.0s |
| `LIST cachedContents` (1 entry) | ~1.7s |
| `generateContent` referencing a cache | ~7.3s |

The `LIST`-by-`displayName` lookup strategy that proposal 008 specifies costs ~1.7s
**on the warm path**, which is the common path. That alone disqualifies it as a
per-request lookup.

### Model support is not uniform

`gemini-3-flash-preview` cannot do explicit caching at all — `cachedContents` returns
`404 "Publisher model ... was not found or your project does not have access to it"`, even
though the same model serves `generateContent` normally. That model is the Pypestream
action-node default (`AIGW_DEFAULT_GEMINI_MODEL`), so **the single most likely caller of
this feature cannot use it.**

### Baseline that already works

Implicit caching earns the same ~90% discount through the gateway today. Validated
previously on both `main` and `candidate`: a 21,146-token prefix yields
`cached_tokens: 20452` (96.7% hit), priced on the `input_cached_tokens: 5e-08` tier —
$0.0015496 vs $0.0107535 undiscounted. Storage is free.

## Where this can physically live in the codebase

The single most important architectural fact, and the one proposal 008 leaves vague:

> **`Translator.RequestBody` cannot do this.** It is a pure function — no `context.Context`,
> no I/O, documented "created per request and is not thread-safe"
> ([`internal/translator/translator.go:46`](../../../internal/translator/translator.go)).
> No cache resolution can happen inside a translator.

The correct and only hook is the **upstream filter**:

```
routerProcessor.ProcessRequestBody     (has ctx, NO backend selected yet -> region unknown)
        |
        v
upstreamProcessor.SetBackend(ctx, backend, ...)      <-- backend, region, project, auth handler known
        |
        v
upstreamProcessor.ProcessRequestHeaders(ctx, ...)    <-- HOOK HERE
        |  1. resolve cache  (new)
        |  2. translator.RequestBody(...)
        |  3. handler.Do(ctx, ...)   -> prepends /v1/projects/{p}/locations/{l}, adds bearer
        v
    upstream Vertex call
```

Why this point and no other
([`internal/extproc/processor_impl.go:344`](../../../internal/extproc/processor_impl.go)):

- It has a `context.Context`.
- It runs **after** backend selection, so `u.backendName`, and the `gcpHandler`'s `region`,
  `projectName` and `tokenSource` are all known — cache identity needs all three.
- It runs **before** `translator.RequestBody`, so the resolved cache name can be injected
  into the outgoing body.
- It **re-runs on retry** (`u.onRetry()`, `u.parent.upstreamFilterCount > 1`), which gives
  correct cache re-resolution on failover to a different region for free.
- The `upstreamProcessor` object **persists across request and response phases**. This
  dissolves proposal 008's open "Request/Response Phase Data Sharing" problem entirely —
  no Envoy dynamic metadata, no custom headers needed. Cache-write token counts recorded in
  the request phase are simply available as struct fields in `ProcessResponseBody`.

Injection into the translator should follow the codebase's existing optional-setter idiom
(`ContentTypeSetter`, `RequestHeadersSetter`, `HeaderValueFilterSetter`, applied at
[`processor_impl.go:716-730`](../../../internal/extproc/processor_impl.go)): add a
`CachedContentSetter` interface implemented only by the Vertex translator.

### The hard ceiling nobody has written down

```go
// internal/extensionserver/post_translate_modify.go:481
extProcConfig.MessageTimeout = durationpb.New(10 * time.Second)
...
FailureModeAllow: false        // :827
```

**The ext_proc message timeout is 10 seconds, hardcoded, and failure mode is closed.**
It is not exposed through `AIGatewayRoute` or any CRD. Every synchronous thing
`ProcessRequestHeaders` does — cache lookup, cache creation, retries, lock waits — comes
out of that 10s budget, and blowing it fails the request rather than passing it through.

This is a *tighter* constraint than the 47s Knative/kourier ceiling in the action-node
environment, and it is the constraint that actually kills synchronous cache creation:
a ~2.5s create is 25% of the budget, a 9.4s create is 94% of it, and the 68% failure rate
means retries are the norm rather than the exception.

## Interface: critique and recommendation

The requested shape has three problems, one of them serious.

### 1. `cacheConfig.prompt` breaks request portability (serious)

`prompt` duplicates content that normally already lives in `messages`. That forces an
unpleasant choice:

- **Caller sends the prompt in both places** → the gateway must strip it from `messages`
  by string-matching, or the model sees it twice and the caller is billed twice. String
  matching against a "long stable prompt" is exactly the kind of fragile that fails
  silently when a caller changes a newline.
- **Caller sends it only in `cacheConfig.prompt`** → the request is no longer a valid,
  self-contained OpenAI request. Any backend without explicit caching — Anthropic, OpenAI,
  Bedrock, *or Gemini after a failed cache create* — receives a request with its system
  prompt missing. The model silently answers without its instructions. That is a worse
  failure than the silent drop this feature is meant to fix.

The second case is unavoidable under fail-open, which means a `cacheConfig.prompt` design
**must** define a canonical reinjection point (which role, which position) for every
non-caching path. The requested interface does not specify one, and any choice the gateway
makes will be wrong for someone.

A pointer into the existing `messages` has none of these problems. The content stays in
exactly one place, the request stays valid and portable, and the fallback path is "do
nothing".

### 2. `use: false` is ambiguous

It can plausibly mean "don't create a cache", "don't read an existing cache", "bypass the
cache for this call but keep it warm", or "invalidate". Callers will assume different ones.
Drop the boolean: **presence of the config means use it.** Add an explicit
`mode: "bypass"` later if a real need appears.

### 3. A boolean is not enough on day one

`ttl` is not optional-in-practice, because storage is billed hourly and any default the
gateway picks is a bill someone did not ask for. `scope` is needed because on Vertex the
cache lives in one project: two tenants sending identical prompt text would otherwise share
a cache resource. The content is identical by definition so this is not a content leak, but
it is a covert channel (tenant B's hit reveals that someone else cached that text) and it
misattributes storage cost to whoever created it first.

### Recommended interface

Use a **breakpoint marker, not a second copy of the text**, and put policy in the CRD
rather than in every request.

Per-request (the part the caller controls):

```json
{
  "model": "gemini-2.5-flash",
  "messages": [
    { "role": "system",
      "content": [{ "type": "text", "text": "<long stable prompt>",
                    "cache_control": { "type": "ephemeral", "ttl": "600s" } }] },
    { "role": "user", "content": "the actual question" }
  ]
}
```

`cache_control` is already in the schema at
[`internal/apischema/openai/openai.go:461`](../../../internal/apischema/openai/openai.go)
and is already honoured by the Anthropic and Bedrock translators. It is proposal 008's
interface. Nothing new needs inventing.

Per-route (the part the platform controls), on `AIGatewayRoute`:

```yaml
promptCache:
  enabled: true
  minTokens: 2048        # don't attempt below this
  ttl: 10m               # clamped [5m, 1h]
  maxTTL: 1h
  scope: tenant          # cache key salt source: none | tenant | header:<name>
  mode: async            # async (recommended) | sync
  failOpen: true         # non-negotiable in practice; see below
```

**And, to satisfy the stated requirement,** accept `cacheConfig` as a thin, explicitly
documented alias that desugars to the above before anything else runs:

```jsonc
"cacheConfig": {
  "prompt": "...",       // if the text is NOT already in messages, prepend it as a
                         // system message; if it IS present verbatim, mark it in place
  "ttl": "600s",         // optional, clamped by route policy
  "scope": "tenant-a"    // optional
}
```

Desugaring rules must be written down and tested, because they are where the ambiguity
goes to hide:

1. If `cacheConfig.prompt` matches an existing message's text content exactly, attach
   `cache_control` to that content part. No duplication.
2. Otherwise, **prepend** it as a `system` message carrying `cache_control`. The request
   is now self-contained and portable, and fail-open works.
3. `use: false` or absent → strip `cacheConfig`, change nothing else.
4. Both `cacheConfig` and `cache_control` present → `400 invalid_cache_config`. Do not
   guess.

This gives callers the field they asked for while keeping one internal representation, one
set of translator changes, and a request that stays valid on every fallback path.

## Reconciliation with proposal 008

Proposal 008 is accepted upstream (PR #1792, merged 2026-03-01) but is **docs and four
SVGs only — no implementation**. Its interface is `cache_control`, which is a different
shape from the `cacheConfig` requirement.

| Dimension | Proposal 008 | This document |
|---|---|---|
| Caller interface | `cache_control` breakpoint | `cache_control` breakpoint, **plus** `cacheConfig` as fork-local sugar |
| Where cache resolution runs | External "Cache Service" microservice | In-process `internal/promptcache`, called from the upstream filter |
| Cache lookup | `LIST cachedContents` + match `displayName` | Two-tier in-memory/Redis; Vertex `LIST` only for cold reconciliation |
| Creation mode | Synchronous; async deferred to "Future Enhancements" | **Async only.** Synchronous is not deployable — see measurements |
| On cache failure | Fail fast, return error to client | **Fail open**, loudly instrumented |
| Request/response data sharing | Listed as an open problem (dynamic metadata / headers / extproc state) | Not a problem — `upstreamProcessor` already persists across phases |
| Cache expiry handling | Not addressed | Must be; stale references return `400`/`404` and hard-fail the completion |

**Where this document adopts 008:** the interface. `cache_control` is the right primitive
and is already half-implemented in the codebase.

**Where it diverges, and what that costs:**

- *Sync → async.* Not a preference. A 10s hardcoded ext_proc budget with `FailureModeAllow:
  false`, against a create API measured at 68% failure and ~2.5s per attempt, makes 008's
  primary mode unshippable. 008's own deferred "Async Caching Mode" section should be
  promoted to the only mode.
- *Fail-fast → fail-open.* 008 argues fail-fast so "users are aware of cache failures
  rather than silently paying full token prices". The intent is right; the mechanism is
  wrong. At a 68% failure rate, fail-fast means failing most cold requests for an
  *optimization*. Awareness belongs in headers, metrics and spans — not in the status code
  of the user's completion.
- *External service → in-process.* 008's cache service is a second network hop (deploy,
  auth, availability) inside a 10s budget, and it needs its own copy of GCP credentials
  that the gateway already holds in `gcpHandler`. The only argument for it is language
  choice.
- *Adding `cacheConfig`.* This is the genuine divergence and it costs upstream-ability —
  see [Upstream-ability](#upstream-ability). Mitigation: keep `cacheConfig` as a pure
  desugaring layer touching one file, so the mechanism underneath stays 008-shaped and
  upstreamable.

## Per-provider mapping

| Provider family | Native mechanism | `cacheConfig` maps to | Gateway owns lifecycle? | Status |
|---|---|---|---|---|
| **GCPVertexAI** (Gemini) | Cache **resource**: `POST .../cachedContents`, referenced by `cachedContent` on `generateContent` | Create/reuse resource; set `cachedContent` | **Yes** — this is the only family with a lifecycle | **Not supported today.** Needs new schema field + new package. Blocked on reliability. |
| **Anthropic** / **GCPAnthropic** / **AWSBedrockAnthropic** | Inline `cache_control: {type: ephemeral}` breakpoints | Attach `cache_control` to the prefix's last content part | **No** — nothing to create or delete; the provider owns it | **Works today.** `cacheConfig` is pure sugar. Min 1,024 tok/block, max 4 breakpoints, `ephemeral` only, prefix-anchored. No 1h TTL documented. |
| **AWSBedrock** (Converse, non-Anthropic) | `cachePoint` blocks | Insert `cachePoint` after the prefix | **No** | Partial. Wiring exists via `anthropicCachePoint`/`appendCachePoint` in `anthropic_awsbedrock.go`. |
| **OpenAI** / **AzureOpenAI** | Automatic prefix caching; `prompt_cache_key` is a *routing hint*, not a cache handle | Nothing faithful. At best pass through `prompt_cache_key` | **No — impossible** | **Not supported, and should say so.** See below. |

### Explicit "not supported" cells

- **OpenAI has no explicit cache to manage.** Prefix caching is automatic and opaque;
  `prompt_cache_key` only influences which server-side shard a prefix lands on. There is no
  create, no id, no TTL, no delete. `use: true` cannot be honoured. The gateway must return
  `x-ai-eg-cache-status: unsupported-provider` and leave the request alone. Inventing a
  lossy mapping here would be worse than the current silent drop. (Upstream #2454 tracks
  `prompt_cache_breakpoint`; it is open and untriaged.)
- **`gemini-3-flash-preview` cannot do explicit caching**, so on that model the answer is
  `unsupported-model` even though the provider family supports the feature. Support must be
  a per-model allowlist, not a per-provider flag.
- **Anthropic cache-write tokens are not retrievable.** Anthropic deliberately does not
  expose them ("tracked internally for billing purposes"), so cache-write cost attribution
  is impossible on that family and the corresponding span attribute must be omitted rather
  than zero-filled.
- **Gemini + `cache_control` is actively harmful.** Previously measured: an identical
  request with `cache_control` on the system block gave `prompt=9488 cached=None`; with it
  removed, `prompt=9488 cached=9188`. Sending the marker *loses* the implicit-cache hit.
  The likely mechanism is that attaching `cache_control` forces the message into
  array-of-parts form, which changes how `openai_gcpvertexai.go` builds Gemini `parts` and
  perturbs the serialized prefix. **Consequence: on Gemini the gateway must strip
  `cache_control` before translation**, whether or not explicit caching is enabled. This is
  a bug worth fixing on its own, independent of this feature.

## Cache identity and reuse

The cache key must include every input that changes what the cached prefix *means* or where
it physically lives. Getting this wrong either never reuses or serves the wrong context.

```
cacheKey = SHA-256(
    "v1"                      // key-schema version; bump to invalidate everything
  | gcpProject                // cache resource lives in one project
  | gcpRegion                 // and one region
  | publisher + "/" + model   // resolved model AFTER modelNameOverride
  | canonicalJSON(systemInstruction)
  | canonicalJSON(toolDeclarations)   // tools are part of the cached prefix
  | canonicalJSON(cachedContentParts) // the prefix up to and including the breakpoint
  | scopeSalt                 // "" | tenant id | header value, per route policy
)
```

Notes on each term:

- **Project and region are mandatory.** A `us-central1` cache is unusable from
  `europe-west4`. They are known at the hook point because `SetBackend` has already run.
- **Model must be post-override.** `u.modelNameOverride` is applied before translation; the
  cache is bound to the model in its create body, and mismatches fail.
- **Tools must be included.** Gemini caches tool declarations as part of the prefix, and
  proposal 008 gets this right. Omitting them would serve a cache built with different tools.
- **Canonical JSON, not raw bytes.** Key off the normalized structure, so incidental
  whitespace and key ordering do not fragment the cache. Conversely, do **not** key off the
  raw request body — it contains the user's question, which changes every request.
- **`scopeSalt` defaults to empty** (share across tenants) only when the route explicitly
  says so. For multi-tenant routes, default to `tenant`.

Store the key in the Vertex `displayName` as `aigw-v1-<first 32 hex of key>`, so an operator
(or a cold-start reconciler) can map a resource back to a key without the gateway's store.

## State

The ext-proc is stateless per request; the mapping `cacheKey -> cacheResourceName` must live
somewhere. Three options, evaluated:

| Option | Warm-path latency | Survives restart | Cross-pod | Verdict |
|---|---|---|---|---|
| **A. In-memory LRU per pod** | ~0ms | No | No | Necessary, insufficient |
| **B. Redis** | ~1ms | Yes | Yes | Necessary for correctness at >1 replica |
| **C. Vertex `LIST` by `displayName`** | **~1.7s** | Yes | Yes | **Disqualified as a request-path lookup** |

Option C is what proposal 008 specifies. At ~1.7s on the *warm* path out of a 10s budget,
for the *common* case, it is not viable. It remains useful as a cold-start reconciliation
sweep, run off the request path.

**Recommended: A backed by B, with C as an offline reconciler.**

```
request -> in-memory LRU  (hit: ~0ms, done)
             miss
              v
           Redis GET      (hit: ~1ms, populate LRU, done)
             miss
              v
        async create job  (request proceeds UNCACHED this time)
```

### On reusing the ratelimit Redis

The cluster already runs `redis-rate-limit` / `envoy-ai-gateway-ratelimit`. Reusing it is
tempting and mostly fine, with two caveats worth stating plainly:

- **Blast radius.** `envoy-ratelimit` is availability-critical for every request. Adding a
  second workload with different access patterns (larger values, different TTLs, occasional
  bursts) couples an optimization to a critical path. At minimum use a **separate logical
  DB** and a distinct key prefix (`aigw:pc:*`); if the feature ever gets real traffic,
  give it its own Redis.
- **Redis must be fail-open.** If Redis is unreachable, the gateway falls back to the
  in-memory LRU and, failing that, to no cache. A cache-metadata store must never be able to
  fail a completion.

### What in-memory means behind an HPA at 1–2 replicas

With `N` replicas and no shared store, a warm cache is only known to the pod that created
it, so steady-state hit rate is roughly `1/N` — 50% at two replicas, and worse during a
scale-out. Every pod that misses starts its own create, so you also pay `N` × storage for
the same content. At N=1 this is invisible in testing and then regresses the moment the HPA
scales. **In-memory alone is not correct at >1 replica**; Redis is what makes it correct.
Note also that ext-proc pods restart on every config rollout, which is frequent.

### Stale entries are not benign

Because a stale reference returns `400`/`404` and **fails the completion**, the store's
entries must be treated as hints, not facts:

- Store `expireTime` alongside the resource name; treat an entry as missing once within 60s
  of expiry (clock skew margin).
- On `400 INVALID_ARGUMENT "Invalid resource state"` or `404 NOT_FOUND ... cached content
  metadata`, **evict the key and retry the completion once without `cachedContent`.** The
  upstream filter already re-runs on retry, so this rides existing machinery — but note the
  retry costs another full completion attempt inside the 10s/47s budgets.

## Concurrency

Two simultaneous first-requests for the same prompt must not create two caches, because
Vertex does not deduplicate (measured: 12 identical creates → 12 billed resources).

**Per-pod:** `golang.org/x/sync/singleflight` keyed on `cacheKey`. Cheap and sufficient
within a pod.

**Cross-pod:** Redis `SET aigw:pc:lock:<key> <pod-id> NX PX 30000`. Loser does not wait —
it proceeds uncached (async mode makes this free) and picks up the cache on a later request.

Failure modes, honestly:

| Failure | Consequence | Mitigation |
|---|---|---|
| Lock holder dies mid-create | Key locked until `PX` expiry | Short `PX` (30s); worst case one duplicate cache |
| Redis unreachable | Degrades to per-pod singleflight | Bounded: at most `N` duplicate caches |
| Create succeeds, Redis write fails | Orphan cache, billed for its full TTL, never used | Short TTL caps the loss; `displayName` lets the reconciler find it |
| **Singleflight shares a flaky failure** | One spurious `400` fails all `N` waiters at once | **Do not share failures.** Share successes only; let each caller fail independently and fall through to uncached |

That last row is specific to this API's 68% failure rate and is easy to get wrong —
`singleflight` shares errors by default.

## Lifecycle

**Who sets TTL:** the caller may request one; the route policy clamps it to
`[5m, maxTTL]`. Never honour an unbounded TTL — storage bills hourly.

**Who refreshes:** Vertex supports `PATCH` on `expireTime`. Sliding refresh on every hit
would add a write per request, so refresh only when remaining TTL drops below 25%, and do it
**asynchronously**, off the request path.

**Who deletes:** *nobody, by default.* Rely on TTL expiry — Vertex reaps the resource
itself. Explicit deletion introduces races (pod A deletes while pod B is mid-request) whose
payoff is small. Instead **cap the TTL low (5–10 minutes)** so the worst-case leak is
bounded by construction:

```
worst-case leaked storage cost = creates_per_hour x TTL_hours x tokens x $1/M-tokens/hour
```

At 10 creates/hour, 10-minute TTL, 20k tokens on Flash: `10 x 0.167 x 0.02 x $1 = $0.033/hour`.
Acceptable. At a 1-hour TTL with a subtly leaky key (a timestamp accidentally in the hash),
the same workload leaks `10 x 1 x 0.02 x $1 = $0.20/hour` and grows — which is why the key
derivation above is versioned and must be unit-tested for determinism.

A **reconciler** (every 15 min, leader-elected) should `LIST` caches whose `displayName`
carries the `aigw-v1-` prefix and delete those absent from the store. This is the only
defence against orphans from crashed pods. It must be leader-elected — concurrent
reconcilers racing deletes against live requests would reintroduce the `400`
"Invalid resource state" failure.

## Latency and the two timeout ceilings

Two budgets, and the tighter one is the one nobody has written down:

| Ceiling | Value | Scope | Configurable? |
|---|---|---|---|
| **ext_proc `MessageTimeout`** | **10s** | Each ext_proc message, incl. all of `ProcessRequestHeaders` | **No — hardcoded** at `post_translate_modify.go:481` |
| ext_proc `GrpcService.Timeout` | 30s | The gRPC stream | No — hardcoded (`:488`) |
| kourier / Knative `REVISION_TIMEOUT_SECONDS` | 47s | Whole action-node request | Per-environment |

Synchronous creation against the 10s budget:

```
cache lookup (Redis)      ~0.001s
cache create              ~2.1-2.8s typical, 9.4s observed
  x retries (68% fail)    ~2.5s each; 1 retry recovers only 1 in 5
translator + auth          ~0.01s
                          ------------------------------
expected cold path         5-10s+, frequently exceeding the budget
```

With `FailureModeAllow: false`, exceeding it **fails the request**. And this is all *before*
the completion call, which itself measured ~7.3s with a cache and must fit under 47s.

**Decision: creation is asynchronous (fire-and-forget). The first request pays full price.**

- Cold-path added latency: **zero**.
- The 68% failure rate becomes a background concern — logged, metered, retried with
  backoff — instead of a user-visible error.
- The 10s budget is never at risk.
- Cost of the choice: the first request (and any request while the create is in flight) does
  not benefit. For a prefix reused hundreds of times this is negligible; for a prefix used
  twice, the feature never pays for itself anyway.

A `mode: sync` option may be offered for `europe-west4`-class regions where creation is
reliable, but it must be off by default and documented with the 10s ceiling.

## Fail-open vs fail-closed

**Fail open, always — but never silently.**

The completion is the product; the cache is an optimization. Failing a user's request
because an optimization failed is the wrong trade at any failure rate, and at a measured 68%
it is indefensible. This is a direct, deliberate reversal of proposal 008's "fail fast, no
silent fallback".

008's underlying concern is correct: users must not silently pay full price. The fix is to
make the outcome *visible*, not to make it *fatal*. Concretely, every request carries its
cache outcome:

| Condition | Behaviour | `x-ai-eg-cache-status` |
|---|---|---|
| Cache hit | `cachedContent` referenced | `hit` |
| Key unknown, create dispatched | Proceed uncached | `miss-creating` |
| Below `minTokens` | Proceed uncached, no create | `miss-too-small` |
| Model can't cache (e.g. `gemini-3-flash-preview`) | Proceed uncached, no create | `unsupported-model` |
| Provider has no explicit cache (OpenAI) | Proceed unchanged | `unsupported-provider` |
| Create failed (the spurious 400, quota, region) | Proceed uncached | `miss-error` |
| Stale reference → `400`/`404` | Evict, retry once uncached | `stale-retried` |
| Redis down | Proceed on LRU or uncached | `degraded` |
| Both `cacheConfig` and `cache_control` sent | **`400 invalid_cache_config`** | — |

The one fail-closed case is a malformed *request*, which is the caller's bug and cheap to
fix. Everything else fails open.

## Region and project affinity

A Vertex cache belongs to exactly one project **and** one region, which is why both are in
the cache key.

This works out cleanly because resolution happens in the upstream filter, *after* routing:

- **Normal routing:** region known before resolution. Correct by construction.
- **Failover / retry:** `ProcessRequestHeaders` re-runs with a new `u.handler` for the new
  region → different `cacheKey` → correct (cold) behaviour in the new region. No stale
  cross-region reference is possible. The cost is a second cold path.
- **Weighted / priority routing across regions:** here be dragons. Storage cost multiplies
  by the number of regions receiving traffic, while hit rate *divides* across them. A route
  split 50/50 across two regions pays 2× storage for ~½ the hits each.

**Anti-pattern to document:** do not enable prompt caching on a route with multi-region
weights unless per-region traffic against the same prefix exceeds the break-even rate
derived below. With a 10-minute TTL and traffic spread thin across regions, it is entirely
possible to pay storage in every region and get hits in none — strictly worse than doing
nothing, because implicit caching would have been free.

## Observability plan

The stated history here is silent drops, so this is the part that must not be cut.

**Response header** (safe — does not break OpenAI-shaped clients):

```
x-ai-eg-cache-status: hit | miss-creating | miss-too-small | miss-error
                    | unsupported-model | unsupported-provider | stale-retried | degraded
x-ai-eg-cache-key: aigw-v1-<32 hex>        # debug builds / opt-in only
```

Do **not** add a non-standard field to the response JSON body — that breaks strict OpenAI
clients. The existing `usage.prompt_tokens_details.cached_tokens` already reports real,
provider-sourced hits and should stay the source of truth for billing.

**Metrics** (new, alongside the existing `genai` meters in `internal/metrics`):

```
gen_ai.gateway.prompt_cache.operations_total{operation,result,model,backend,region}
    operation = lookup | create | refresh | evict
    result    = hit | miss | created | failed | stale
gen_ai.gateway.prompt_cache.create_duration_seconds{model,region}   histogram
gen_ai.gateway.prompt_cache.entries{region}                          gauge
gen_ai.gateway.prompt_cache.tokens_written_total{model,region}       counter  # storage cost driver
```

The `create{result=failed}` counter is what makes the 68% failure rate visible in
production instead of being discovered by measurement a year later. Alert on it.

**Tracing.** Both `main` and `candidate` run `AI_GATEWAY_TRACING_SEMCONV=gen_ai`. The
plumbing already exists: `TokenUsage` carries `cachedInputTokens` and
`cacheCreationInputTokens` (`internal/metrics/metrics.go:150-153`), and
`otelgenai/endpoints.go:378` already emits cache-read/cache-creation attributes. Add to the
existing span rather than creating a child span:

```
gen_ai.gateway.prompt_cache.status      = hit | miss-creating | ...
gen_ai.gateway.prompt_cache.key         = aigw-v1-<32 hex>
gen_ai.gateway.prompt_cache.resource    = <vertex resource name>      # hit only
gen_ai.gateway.prompt_cache.ttl_s       = 600
gen_ai.gateway.prompt_cache.error       = <message>                   # failures only
```

Because the `upstreamProcessor` persists across phases, request-phase cache facts are simply
available when the response-phase span attributes are written — no dynamic metadata needed.

For OpenInference, map `hit` to the existing `LLMTokenCountPromptCacheHit`
(`internal/tracing/openinference/openai/response_attrs.go:66`) and leave the rest to the
gen_ai attributes.

**Also worth fixing independently:** `geminiUsageToOpenAIUsage` currently drops
`trafficType`, `promptTokensDetails[]`, `cacheTokensDetails[]` and
`candidatesTokensDetails[]`. `trafficType` in particular distinguishes provisioned
throughput from pay-as-you-go and matters for cost attribution.

## Cost model

Prices (Vertex, Flash-class, Sept 2026). Cache read at `5e-08`/token is verified from our
own billing tier; the storage figure is from public pricing summaries and should be
confirmed against an actual invoice before anyone commits to it.

| Meter | Rate |
|---|---|
| Standard input | ~$0.50 / M tokens (`5e-07`/token) |
| Cached input (read) — implicit **and** explicit | **$0.05 / M tokens** (`5e-08`/token, verified) |
| **Explicit cache storage** | **~$1.00 / M tokens / hour** (Flash) · ~$4.50 (Pro) |
| Implicit cache storage | **$0.00** |

The decisive point: **explicit and implicit caching earn the identical read discount.**
Explicit adds a storage meter and buys nothing on price.

### Break-even

For a prefix of `T` tokens at `R` requests/hour with implicit-cache miss rate `m`:

```
explicit extra cost   = T x $1/M per hour                        (storage)
explicit extra saving = R x m x T x (5e-07 - 5e-08)              (misses that now hit)

worth it when:   R x m x T x 4.5e-07  >  T x 1e-06
                 R x m               >  2.22
```

`T` cancels — **size does not matter, only the rate of implicit-cache misses does.**

> **Explicit caching pays for itself only above ~2.2 implicit-cache-missing requests per
> hour against the same prefix, in the same region, on the same model.**

### Applying it to the measured baseline

Measured implicit hit rate on the 21,146-token prefix: `20452/21146 = 96.7%`, so `m ≈ 0.033`.

```
R x 0.033 > 2.22   =>   R > 67 requests/hour   (~1.1 req/min, sustained, same prefix+region)
```

Below that, explicit caching **loses money** relative to doing nothing. Concretely, for a
20k-token prefix at 10 req/hour:

| | Cost / hour |
|---|---|
| Implicit only (today) | 10 × 20k × (0.967×5e-08 + 0.033×5e-07) = **$0.0130** |
| Explicit, 100% hit | 10 × 20k × 5e-08 + 0.02 × $1 = $0.0100 + **$0.0200 storage** = **$0.0300** |

Explicit is **2.3× more expensive** at that rate. And that ignores the cache-write cost of
the initial create and the 68% retry amplification.

### When it is actually worth using

All of these must hold:

1. Sustained > ~67 req/hour against one prefix, one region, one model, **and**
2. implicit caching demonstrably misses (measure `cached_tokens` first — do not assume), **and**
3. the model supports explicit caching (`gemini-2.5-flash` yes, `gemini-3-flash-preview` no), **and**
4. the region's create path is reliable (`europe-west4` yes, `us-central1` currently no), **and**
5. the caller genuinely benefits from *not resending* the context — bandwidth, or a request
   body approaching a size limit. This is the one advantage implicit caching cannot match,
   and it is the only honest reason to reach for explicit caching at low volume.

No current Pypestream workload is known to satisfy 1–4 simultaneously.

## File-by-file change inventory

Scoped to the **full** feature, so the cost is visible. The recommended subset is a small
fraction of it (marked ✅ = in the recommended subset, ⏸ = deferred).

### New package: `internal/promptcache/`

| File | Purpose | Size |
|---|---|---|
| ⏸ `cache.go` | `Resolver` interface; `Resolve(ctx, key) (name, status, error)`; async create dispatch | ~250 |
| ⏸ `key.go` | Canonical key derivation; must be deterministic across Go map ordering | ~120 |
| ⏸ `store.go` | Two-tier LRU + Redis; fail-open on Redis errors | ~200 |
| ⏸ `vertex.go` | `cachedContents` create/get/patch/delete; **the `.../publishers/google/models/{m}` model-path form**; classify the `400`/`404` stale signatures | ~250 |
| ⏸ `singleflight.go` | Per-pod singleflight + Redis `SET NX PX`; **must not share failures** | ~120 |
| ⏸ `reconciler.go` | Leader-elected orphan sweep by `displayName` prefix | ~150 |
| ⏸ `*_test.go` | Table tests + a fake Vertex that reproduces the spurious 400 | ~700 |

### `internal/apischema/`

| File | Change | ✅/⏸ |
|---|---|---|
| `gcp/gcp.go` | Add `CachedContent string \`json:"cachedContent,omitempty"\`` to `GenerateContentRequest` (currently absent — the root cause of the silent drop) | ✅ |
| `openai/openai.go` | Add `CachedContent` to `GCPVertexAIVendorFields` (008's explicit-cache-name case) | ✅ |
| `openai/openai.go` | Add `CacheConfig *CacheConfig` to `ChatCompletionRequest` + the type | ⏸ |

`google.golang.org/genai@v1.62.0` already provides `CachedContent`, `CreateCachedContentConfig`
and `CachedContentUsageMetadata`, so no hand-rolled types are needed for the Vertex calls.

### `internal/translator/`

| File | Change | ✅/⏸ |
|---|---|---|
| `translator.go` | New optional `CachedContentSetter` interface (mirrors `ContentTypeSetter`) | ⏸ |
| `openai_gcpvertexai.go` | `applyVendorSpecificFields`: honour `CachedContent`; implement `CachedContentSetter`; **strip `cache_control` before translation** (it destroys implicit hits) | ✅ (strip) / ⏸ (setter) |
| `gemini_helper.go` | Surface `trafficType`, `cacheTokensDetails[]`; populate `cacheCreationInputTokens` | ✅ |
| `anthropic_helper.go` | Desugar `cacheConfig` → `cache_control` for the Anthropic family | ⏸ |
| `openai_gcpvertexai_test.go`, `anthropic_*_test.go` | Coverage incl. the both-fields-set `400` | ✅/⏸ |

### `internal/extproc/`

| File | Change | ✅/⏸ |
|---|---|---|
| `processor_impl.go` | `SetBackend` (line 680): wire the resolver, apply `CachedContentSetter` alongside the existing setters (~line 716) | ⏸ |
| `processor_impl.go` | `ProcessRequestHeaders`: resolve before `translator.RequestBody` (line 364) | ⏸ |
| `processor_impl.go` | `ProcessRequestHeaders`: `cacheConfig` desugaring + `400 invalid_cache_config` | ⏸ |
| `processor_impl.go` | `ProcessResponseHeaders`: emit `x-ai-eg-cache-status` | ✅ |
| `processor_impl.go` | `ProcessResponseBody`: detect the stale `400`/`404`, evict, retry uncached | ⏸ |
| `server.go` | Construct the resolver; wire Redis config | ⏸ |
| `mocks_test.go`, `processor_impl_test.go` | Mock resolver; stale-reference and fail-open tests | ⏸ |

### `internal/backendauth/`

| File | Change | ✅/⏸ |
|---|---|---|
| `gcp.go` | Expose `region`, `projectName`, and a token accessor. Today `gcpHandler` holds all three privately and only `Do()` uses them; the resolver needs them without re-reading credentials | ⏸ |
| `../filterapi/runtime.go` | Widen `BackendAuthHandler`, or add a narrow `GCPCoordinates` interface asserted at the call site (preferred — avoids touching every handler) | ⏸ |

### API / CRD

| File | Change | ✅/⏸ |
|---|---|---|
| `api/v1alpha1/api.go` | `PromptCacheSpec` on `AIGatewayRoute` | ⏸ |
| `api/v1alpha1/zz_generated.deepcopy.go` | Regenerate | ⏸ |
| `manifests/charts/ai-gateway-crds-helm/**` | Regenerate CRDs | ⏸ |
| `internal/filterapi/filterconfig.go` | Propagate to `Backend`/`Config` | ⏸ |
| `internal/controller/**` | Translate CRD → filter config | ⏸ |

### Metrics / tracing

| File | Change | ✅/⏸ |
|---|---|---|
| `internal/metrics/metrics.go` | New instruments; `RecordPromptCacheOp` | ✅ |
| `internal/tracing/otelgenai/endpoints.go` | New span attributes | ✅ |
| `internal/tracing/openinference/openai/response_attrs.go` | Map hit → `LLMTokenCountPromptCacheHit` | ✅ |

### Docs / tests

| File | Change | ✅/⏸ |
|---|---|---|
| `site/docs/capabilities/llm-integrations/prompt-caching.md` | Document `cacheConfig`, the per-provider table, the **cost model and break-even** | ✅ |
| `site/docs/capabilities/llm-integrations/vendor-specific-fields.md` | Document `cachedContent` | ✅ |
| `tests/extproc/**`, `tests/e2e/**` | Cover fail-open, stale reference, unsupported model | ⏸ |
| `internal/testupstream/**` | Fake `cachedContents` endpoint | ⏸ |

**Rough sizing.** Recommended subset (✅): ~400–600 lines across ~12 files, 2–4 days
including tests. Full feature (⏸): ~2,500–3,500 lines across ~30 files, plus CRD and
controller work, plus a Redis dependency and a reconciler to operate — realistically 3–5
weeks including e2e, and it inherits a 68%-failure upstream API.

## Upstream-ability

| Path | Effort | Probability | Notes |
|---|---|---|---|
| **Upstream `cacheConfig`** | High | **Low** | Contradicts accepted proposal 008's interface. Would need a new proposal to supersede a merged one. |
| **Upstream the mechanism under 008's `cache_control`** | High | **Medium** | 008 is accepted and unimplemented — an implementation is welcome in principle. But our design reverses two of its explicit decisions (sync→async, fail-fast→fail-open), so it is a proposal amendment, not just code. |
| **Upstream `cachedContent` pass-through only** | **Low** | **Medium-High** | Small, additive, matches 008's "Explicit Cache Name" section verbatim. The obvious first PR. |
| **Fork-only** | Low | n/a | Merge cost on every upstream sync, concentrated in `openai_gcpvertexai.go` and `processor_impl.go` — both actively changed upstream. |

Signals on upstream appetite, none of them encouraging:

- **#1792** (proposal 008) merged 2026-03-01 as docs + 4 SVGs. Six months later there is
  still no implementation — the design was accepted, but nobody is building it.
- **#2528** (Anthropic→Bedrock silently drops `cache_control`) was closed `NOT_PLANNED` the
  same day. That is the *same failure class* as this feature — a caching directive dropped
  in translation — and it was declined immediately.
- **#220** (umbrella, names GCP context caching) and **#2454** (OpenAI
  `prompt_cache_breakpoint`) are both open and untriaged.
- **#1961** (open PR, Gemini native endpoints via prefix-based path dispatch) adds a general
  `RegisterPrefix` mechanism. If a cache-management endpoint is ever wanted, it should ride
  that rather than inventing new path handling — worth tracking regardless of this feature.
- The upstream repo was **renamed**: `envoyproxy/ai-gateway` → `theagentrouter/agent-router`.
  Git and API calls follow the redirect, but GitHub *search* only answers to the new name;
  searching the old name silently returns nothing. Use the new name for any issue/PR/code
  search.

**Recommendation: fork-only for `cacheConfig`; upstream the `cachedContent` pass-through
now as a standalone PR.** Structure the code so `cacheConfig` is a desugaring layer that
touches one file, keeping the mechanism underneath 008-shaped in case upstream appetite
ever appears.

## Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Vertex create is 68% unreliable in `us-central1`** | **High** | Async creation; retry with backoff; alert on `create{result=failed}`. Root cause is Google's, not ours — we cannot fix it. |
| 2 | **Stale references hard-fail completions** (`400`/`404`) | **High** | Expiry margin, evict-and-retry, short TTLs. Still a user-visible failure mode on a losing race. |
| 3 | **Feature is unusable on the default model** (`gemini-3-flash-preview`) | **High** | Per-model allowlist; `unsupported-model` status. No workaround — the model just cannot do it. |
| 4 | **Negative ROI below ~67 req/hour/prefix** | **High** | Document break-even; default off; require measured implicit-miss evidence before enabling. |
| 5 | Multi-region routing multiplies storage, divides hit rate | Medium | Document the anti-pattern; consider refusing to enable on weighted multi-region routes. |
| 6 | Cross-tenant cache sharing (covert channel + cost misattribution) | Medium | `scope: tenant` default on multi-tenant routes. |
| 7 | `cacheConfig.prompt` makes requests non-portable on fallback | Medium | Desugaring rule 2 (prepend as system message) keeps the request self-contained. |
| 8 | Redis coupling to `envoy-ratelimit` | Medium | Separate logical DB + key prefix; fail-open; separate instance if traffic grows. |
| 9 | Orphaned caches leak spend | Medium | Short TTL caps it by construction; leader-elected reconciler. |
| 10 | Merge conflicts with upstream on every sync | Medium | Confine to a desugaring layer + a new package. |
| 11 | 10s ext_proc budget is hardcoded and invisible | Medium | Async design avoids it entirely. Worth documenting for everyone regardless. |
| 12 | **`cache_control` on Gemini destroys implicit hits** | **High, and live today** | Strip it before Gemini translation. **This is a bug now, independent of this feature.** |

## Recommendation

**Do not build gateway-owned explicit prompt-cache lifecycle.**

The feature as specified is a Gemini-only capability presented as cross-provider; on the one
provider where it is real, it is currently 68% unreliable in the region we use, unavailable
on the model our main caller uses, and — by the arithmetic above — *more expensive than
doing nothing* below ~67 requests/hour against the same prefix. Implicit caching already
delivers the identical 90% read discount at 96.7% hit rate with zero storage cost and zero
new moving parts.

**Do build, in this order:**

1. **Fix the live bug (P0, ~half a day).** Strip `cache_control` before Gemini translation.
   Today, a caller who sends the Anthropic-style marker to a Gemini backend *loses* their
   implicit-cache discount — measured `cached=9188` → `cached=None`. This is an active
   regression for anyone doing the reasonable thing.
2. **Make caching visible (P1, ~2 days).** `x-ai-eg-cache-status` header, the prompt-cache
   metrics, the span attributes, and the dropped `usageMetadata` fields. This addresses the
   real, recurring complaint — silent behaviour — and is worth doing whether or not anything
   else is ever built.
3. **`cachedContent` pass-through (P1, ~half a day).** Add the field to
   `gcp.GenerateContentRequest` and `GCPVertexAIVendorFields`. Unblocks callers who want to
   manage their own caches, satisfies proposal 008's "Explicit Cache Name" case, and is the
   obvious upstream PR.
4. **`cacheConfig` → `cache_control` desugaring for the Anthropic family (P2, ~1 day).**
   Gives callers the requested field shape where the underlying capability genuinely works
   today at zero storage cost.
5. **Revisit gateway-owned Vertex lifecycle only when** Google's create reliability is fixed
   (retest with the scripts referenced here), **and** a real workload is measured missing
   implicit cache at > ~67 req/hour/prefix, **and** it runs on a model that supports explicit
   caching.

If a caller needs explicit caching *right now*, the honest answer is item 3 plus
`gemini_explicit_cache.py`: let them own the cache and pass the resource name. That is
roughly half a day of gateway work instead of several weeks, and it puts the flakiness where
someone can see and retry it.

## What could not be verified

- **Whether the 68% failure rate is project-specific.** All measurements are against
  `admin-295215`. It could be a quota or backend-capacity condition on that project rather
  than a general Vertex property. `europe-west4` was clean (12/12 across two runs) from the
  *same* project, which argues for a regional serving issue, but a second project would
  settle it. **Retest before acting on this number.**
- **Cache storage pricing.** The ~$1.00/M-tokens/hour Flash figure comes from public pricing
  summaries, not from a Google Cloud pricing page read directly (the page truncated on
  fetch) and not from an invoice. The read rate (`5e-08`/token) *is* verified from our own
  billing tier. Confirm storage against a real bill before committing to the cost model.
- **Anthropic-family and Bedrock behaviour end-to-end** was read from the code
  (`anthropic_helper.go`, `anthropic_awsbedrock.go`) and existing docs, not exercised
  against live providers in this investigation.
- **The mechanism behind `cache_control` breaking Gemini implicit hits.** The
  array-of-parts hypothesis in [Per-provider mapping](#per-provider-mapping) is inferred
  from reading `openai_gcpvertexai.go`, not proven. The *effect* is measured; the *cause*
  is not.
- **Redis behaviour under this load** was not tested; the fail-open design is on paper.
- **Nothing was measured in prod.** There is no prod AWS access from this workstation and
  none was attempted. All cluster work used the `main`/`candidate` contexts, and all Vertex
  work used the non-prod `admin-295215` project.

### Reproducing the measurements

The 26 cache resources created during this investigation were all deleted; `cachedContents`
in `us-central1`, `us-east1` and `europe-west4` were verified empty afterwards. To re-run,
adapt `/Users/shababqaisar/pypestream/gemini_explicit_cache.py` — the essential loop is
N sequential creates of one fixed >1,024-token payload, counting non-200s. **Delete what
you create**; every resource bills storage for its full TTL whether used or not.
