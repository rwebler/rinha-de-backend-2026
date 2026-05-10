FROM --platform=linux/amd64 rust:1.95-slim AS builder
WORKDIR /app

COPY Cargo.toml ./
COPY src ./src
COPY resources ./resources

RUN cargo build --release --bin build_artifacts --bin server
RUN ./target/release/build_artifacts --input resources/references.json.gz --output /artifacts --clusters 2048 --probes 12

FROM --platform=linux/amd64 debian:bookworm-slim
WORKDIR /app

RUN useradd -r -u 10001 appuser

COPY --from=builder /app/target/release/server /app/server
COPY --from=builder /artifacts /app/artifacts
COPY resources/normalization.json /app/resources/normalization.json
COPY resources/mcc_risk.json /app/resources/mcc_risk.json

RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 9999

ENV BIND_ADDR=0.0.0.0:9999
ENV ARTIFACT_DIR=/app/artifacts
ENV NORMALIZATION_PATH=/app/resources/normalization.json
ENV MCC_RISK_PATH=/app/resources/mcc_risk.json

CMD ["/app/server"]
