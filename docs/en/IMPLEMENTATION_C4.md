# Implementation C4

This document describes the current Rust implementation added in this repository, not the generic competition topology. It focuses on the concrete runtime services, the internal components of the API, and the offline artifact pipeline that prepares the vector-search data.

Relevant source files:

- [`docker-compose.yml`](../../docker-compose.yml)
- [`nginx.conf`](../../nginx.conf)
- [`src/bin/server.rs`](../../src/bin/server.rs)
- [`src/lib.rs`](../../src/lib.rs)
- [`src/search.rs`](../../src/search.rs)
- [`src/bin/build_artifacts.rs`](../../src/bin/build_artifacts.rs)

## Level 1: System Context

```mermaid
flowchart LR
    card[Card authorization system<br/>or load-test client]
    service[Fraud detection backend<br/>Rust + nginx]
    dataset[Static reference data<br/>normalization.json<br/>mcc_risk.json<br/>references.json.gz]

    card -->|GET /ready<br/>POST /fraud-score| service
    dataset -->|offline preprocessing| service
    service -->|JSON: approved + fraud_score| card
```

### Notes

- The backend is an isolated fraud-scoring system. It receives transaction payloads and returns a decision.
- The large reference dataset is not queried as raw JSON at request time. It is transformed into compact artifacts before the API starts serving traffic.

## Level 2: Container Diagram

```mermaid
flowchart LR
    client[Client / k6 / card system]

    subgraph compose[docker-compose topology]
        lb[nginx load balancer<br/>port 9999<br/>round-robin only]
        api1[API instance 1<br/>Rust axum service]
        api2[API instance 2<br/>Rust axum service]
        artifacts[(Packed search artifacts<br/>meta.json<br/>centroids.bin<br/>vectors.bin<br/>labels.bin)]
        config[(Runtime config files<br/>normalization.json<br/>mcc_risk.json)]
    end

    client -->|HTTP| lb
    lb -->|proxy| api1
    lb -->|proxy| api2
    artifacts -->|memory-map on startup| api1
    artifacts -->|memory-map on startup| api2
    config -->|load on startup| api1
    config -->|load on startup| api2
```

### Notes

- `nginx` performs no business logic. It only forwards requests to the two upstream API containers.
- Each API instance loads the same read-only artifact set and answers requests independently.
- The API containers do not depend on an external database, cache, or vector store in the hot path.

## Level 3: API Component Diagram

```mermaid
flowchart TD
    req[HTTP request]
    ready[Readiness handler<br/>GET /ready]
    score[Fraud-score handler<br/>POST /fraud-score]
    parse[Request parser<br/>serde_json]
    vectorize[Vectorization module<br/>14-dim deterministic mapping]
    engine[Search engine<br/>IVF-style clustered scan]
    topk[Top-5 aggregator<br/>fixed-size nearest set]
    decision[Decision module<br/>fraud_count / 5<br/>threshold 0.6]
    fallback[Fallback scorer<br/>always returns HTTP 200]
    resp[HTTP JSON response]

    req --> ready
    req --> score
    score --> parse
    parse -->|valid payload| vectorize
    parse -->|invalid payload| fallback
    vectorize -->|vector ok| engine
    vectorize -->|error| fallback
    engine --> topk
    engine -->|search error| fallback
    topk --> decision
    decision --> resp
    fallback --> resp
```

### Component responsibilities

- **Request parser**: deserializes the incoming JSON body into the Rust DTOs.
- **Vectorization module**: applies the exact 14-dimension mapping from the challenge rules, including UTC hour/day extraction, `-1` sentinels for missing last-transaction fields, clamping, and MCC fallback.
- **Search engine**: quantizes the request vector, ranks coarse centroids, probes a bounded number of inverted lists, and computes squared Euclidean distance over packed vectors.
- **Top-5 aggregator**: maintains the current nearest five candidates without allocating a large sortable structure.
- **Decision module**: converts the five labels into `fraud_score` and `approved`.
- **Fallback scorer**: returns valid JSON on degraded paths so the system avoids non-200 responses during scoring.

## Level 4: Artifact Build Pipeline

```mermaid
flowchart LR
    raw[references.json.gz]
    build[build_artifacts binary]
    stream[Streaming JSON reader]
    quantize[Quantizer<br/>14-dim f32 to 14-dim i8]
    cluster[K-means style coarse clustering]
    assign[Cluster assignment + reorder]
    write[Artifact writer]
    out[(meta.json<br/>centroids.bin<br/>vectors.bin<br/>labels.bin)]

    raw --> build
    build --> stream
    stream --> quantize
    quantize --> cluster
    cluster --> assign
    assign --> write
    write --> out
```

### Notes

- The builder streams the gzipped reference array and does not require a raw expanded JSON file in the runtime image.
- Vectors are quantized to signed bytes so the API can store and scan them more compactly.
- Reordered per-cluster storage keeps each inverted list contiguous, which makes probe scans sequential and cache-friendlier.

## Request Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant N as nginx
    participant A as Rust API
    participant S as Search engine

    C->>N: POST /fraud-score
    N->>A: proxied request
    A->>A: parse JSON
    A->>A: vectorize to 14 dims
    A->>S: score(vector)
    S->>S: rank centroids
    S->>S: scan probe lists
    S-->>A: top-5 labels
    A->>A: compute fraud_score
    A-->>N: 200 JSON
    N-->>C: 200 JSON
```

## Design Intent

- Keep the request path self-contained and read-only after startup.
- Move heavy dataset work into an offline build step.
- Prefer valid `200` JSON responses over surfacing request-path errors.
- Keep the runtime topology compliant with the competition requirement of one load balancer plus two API instances.

[← English README](./README.md)
