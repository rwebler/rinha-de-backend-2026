use std::{env, fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result, anyhow};
use rinha_backend_2026::{
    FraudRequest, load_mcc_risk, load_normalization, search::SearchEngine, vectorize,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Cli {
    input: PathBuf,
    artifact_dir: PathBuf,
    normalization: PathBuf,
    mcc_risk: PathBuf,
    limit: Option<usize>,
    print_mismatches: usize,
    expected_format: ExpectedFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedFormat {
    Auto,
    Approved,
    Label,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetectionKind {
    TruePositive,
    TrueNegative,
    FalsePositive,
    FalseNegative,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct ConfusionMatrix {
    true_positive_detections: u64,
    true_negative_detections: u64,
    false_positive_detections: u64,
    false_negative_detections: u64,
}

#[derive(Default, Debug)]
struct EvalStats {
    matrix: ConfusionMatrix,
    parse_errors: u64,
    search_errors: u64,
}

#[derive(Debug, Serialize)]
struct EvalOutput {
    total: usize,
    evaluated: u64,
    true_positive_detections: u64,
    true_negative_detections: u64,
    false_positive_detections: u64,
    false_negative_detections: u64,
    #[serde(rename = "weighted_errors_E")]
    weighted_errors_e: u64,
    #[serde(rename = "weighted_errors_E_with_errors_as_fn")]
    weighted_errors_e_with_errors_as_fn: u64,
    failure_rate: f64,
    error_rate_epsilon: f64,
    accuracy: f64,
    fraud_recall: f64,
    fraud_precision: f64,
    legit_recall: f64,
    elapsed_ms: f64,
    avg_us_per_record: f64,
    artifact_dir: String,
    probe_count: usize,
    cluster_count: usize,
    adaptive_probing_enabled: bool,
    adaptive_initial_probes: usize,
    parse_errors: u64,
    search_errors: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Label {
    Fraud,
    Legit,
}

fn main() -> Result<()> {
    let cli = Cli::parse(env::args().skip(1))?;
    let raw = fs::read_to_string(&cli.input)
        .with_context(|| format!("failed to read {}", cli.input.display()))?;
    let records = parse_records(&raw).context("failed to parse input as JSON or JSON lines")?;
    let total = records.len();
    let records_to_process = cli.limit.map_or(total, |limit| total.min(limit));

    let normalization = load_normalization(&cli.normalization)?;
    let mcc_risk = load_mcc_risk(&cli.mcc_risk)?;
    let probe_override = env::var("PROBE_COUNT")
        .ok()
        .and_then(|value| value.parse().ok());
    let engine = SearchEngine::open(&cli.artifact_dir, probe_override)?;

    let start = Instant::now();
    let mut stats = EvalStats::default();
    let mut mismatches_printed = 0_usize;

    for (index, record) in records.iter().take(records_to_process).enumerate() {
        let extracted = match extract_case(record, cli.expected_format) {
            Ok(extracted) => extracted,
            Err(error) => {
                stats.parse_errors += 1;
                eprintln!(
                    "{}",
                    json!({
                        "index": index,
                        "error": error.to_string(),
                        "kind": "parse_error"
                    })
                );
                continue;
            }
        };

        let request_json = match serde_json::to_string(&extracted.request) {
            Ok(request_json) => request_json,
            Err(error) => {
                stats.parse_errors += 1;
                eprintln!(
                    "{}",
                    json!({
                        "index": index,
                        "error": error.to_string(),
                        "kind": "parse_error"
                    })
                );
                continue;
            }
        };
        let request: FraudRequest<'_> = match serde_json::from_str(&request_json) {
            Ok(request) => request,
            Err(error) => {
                stats.parse_errors += 1;
                eprintln!(
                    "{}",
                    json!({
                        "index": index,
                        "error": error.to_string(),
                        "kind": "parse_error"
                    })
                );
                continue;
            }
        };

        let vector = match vectorize(&request, &normalization, &mcc_risk) {
            Ok(vector) => vector,
            Err(error) => {
                stats.parse_errors += 1;
                eprintln!(
                    "{}",
                    json!({
                        "index": index,
                        "id": request.id,
                        "error": error.to_string(),
                        "kind": "parse_error"
                    })
                );
                continue;
            }
        };

        let response = match engine.score(&vector) {
            Ok(response) => response,
            Err(error) => {
                stats.search_errors += 1;
                eprintln!(
                    "{}",
                    json!({
                        "index": index,
                        "id": request.id,
                        "error": error.to_string(),
                        "kind": "search_error"
                    })
                );
                continue;
            }
        };

        let kind = stats
            .matrix
            .update(extracted.expected_approved, response.approved);
        if matches!(
            kind,
            DetectionKind::FalsePositive | DetectionKind::FalseNegative
        ) && mismatches_printed < cli.print_mismatches
        {
            mismatches_printed += 1;
            eprintln!(
                "{}",
                json!({
                    "index": index,
                    "id": request.id,
                    "expected_approved": extracted.expected_approved,
                    "predicted_approved": response.approved,
                    "fraud_score": response.fraud_score,
                    "kind": match kind {
                        DetectionKind::FalsePositive => "FP",
                        DetectionKind::FalseNegative => "FN",
                        DetectionKind::TruePositive | DetectionKind::TrueNegative => unreachable!(),
                    }
                })
            );
        }
    }

    let elapsed = start.elapsed();
    let output = build_output(
        total,
        &cli,
        &engine,
        &stats,
        elapsed.as_secs_f64() * 1_000.0,
    );
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

impl Cli {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut input = None;
        let mut artifact_dir = None;
        let mut normalization = None;
        let mut mcc_risk = None;
        let mut limit = None;
        let mut print_mismatches = 0_usize;
        let mut expected_format = ExpectedFormat::Auto;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = Some(PathBuf::from(next_arg(&mut args, "--input")?)),
                "--artifact-dir" => {
                    artifact_dir = Some(PathBuf::from(next_arg(&mut args, "--artifact-dir")?));
                }
                "--normalization" => {
                    normalization = Some(PathBuf::from(next_arg(&mut args, "--normalization")?));
                }
                "--mcc-risk" => mcc_risk = Some(PathBuf::from(next_arg(&mut args, "--mcc-risk")?)),
                "--limit" => limit = Some(parse_usize(next_arg(&mut args, "--limit")?, "--limit")?),
                "--print-mismatches" => {
                    print_mismatches = parse_usize(
                        next_arg(&mut args, "--print-mismatches")?,
                        "--print-mismatches",
                    )?;
                }
                "--expected-format" => {
                    expected_format =
                        ExpectedFormat::parse(&next_arg(&mut args, "--expected-format")?)?;
                }
                "--help" | "-h" => return Err(anyhow!(usage())),
                other => return Err(anyhow!("unknown argument {other}\n{}", usage())),
            }
        }

        Ok(Self {
            input: input.ok_or_else(|| anyhow!("--input is required\n{}", usage()))?,
            artifact_dir: artifact_dir
                .unwrap_or_else(|| env_path("ARTIFACT_DIR", "docker/artifacts")),
            normalization: normalization
                .unwrap_or_else(|| env_path("NORMALIZATION_PATH", "resources/normalization.json")),
            mcc_risk: mcc_risk
                .unwrap_or_else(|| env_path("MCC_RISK_PATH", "resources/mcc_risk.json")),
            limit,
            print_mismatches,
            expected_format,
        })
    }
}

impl ExpectedFormat {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "auto" => Ok(Self::Auto),
            "approved" => Ok(Self::Approved),
            "label" => Ok(Self::Label),
            _ => Err(anyhow!(
                "invalid --expected-format {raw}; expected auto, approved, or label"
            )),
        }
    }
}

impl ConfusionMatrix {
    fn update(&mut self, expected_approved: bool, predicted_approved: bool) -> DetectionKind {
        match (expected_approved, predicted_approved) {
            (false, false) => {
                self.true_positive_detections += 1;
                DetectionKind::TruePositive
            }
            (true, true) => {
                self.true_negative_detections += 1;
                DetectionKind::TrueNegative
            }
            (true, false) => {
                self.false_positive_detections += 1;
                DetectionKind::FalsePositive
            }
            (false, true) => {
                self.false_negative_detections += 1;
                DetectionKind::FalseNegative
            }
        }
    }

    fn evaluated(&self) -> u64 {
        self.true_positive_detections
            + self.true_negative_detections
            + self.false_positive_detections
            + self.false_negative_detections
    }

    fn weighted_errors(&self) -> u64 {
        self.false_positive_detections + 3 * self.false_negative_detections
    }
}

#[derive(Debug)]
struct ExtractedCase {
    expected_approved: bool,
    request: Value,
}

fn extract_case(record: &Value, expected_format: ExpectedFormat) -> Result<ExtractedCase> {
    let expected_approved = extract_expected_approved(record, expected_format)
        .ok_or_else(|| anyhow!("record does not contain a supported expected label field"))?;
    let request = record
        .get("request")
        .cloned()
        .unwrap_or_else(|| request_from_flat_record(record));

    Ok(ExtractedCase {
        expected_approved,
        request,
    })
}

fn request_from_flat_record(record: &Value) -> Value {
    let mut request = record.clone();
    if let Value::Object(ref mut object) = request {
        for key in ["expected", "label", "expected_approved", "approved"] {
            object.remove(key);
        }
    }
    request
}

fn extract_expected_approved(record: &Value, format: ExpectedFormat) -> Option<bool> {
    match format {
        ExpectedFormat::Auto => {
            extract_approved_fields(record).or_else(|| extract_label_fields(record))
        }
        ExpectedFormat::Approved => extract_approved_fields(record),
        ExpectedFormat::Label => extract_label_fields(record),
    }
}

fn extract_approved_fields(record: &Value) -> Option<bool> {
    record
        .get("expected_approved")
        .and_then(Value::as_bool)
        .or_else(|| record.get("approved").and_then(Value::as_bool))
        .or_else(|| {
            record
                .get("expected")
                .and_then(|expected| expected.get("approved"))
                .and_then(Value::as_bool)
        })
}

fn extract_label_fields(record: &Value) -> Option<bool> {
    record
        .get("expected")
        .and_then(parse_label_approved)
        .or_else(|| record.get("label").and_then(parse_label_approved))
}

fn parse_label_approved(value: &Value) -> Option<bool> {
    let label = value.as_str()?;
    let label: Label = serde_json::from_value(Value::String(label.to_ascii_lowercase())).ok()?;
    Some(matches!(label, Label::Legit))
}

fn parse_records(raw: &str) -> Result<Vec<Value>> {
    match serde_json::from_str::<Value>(raw) {
        Ok(value) => unwrap_records(value),
        Err(json_error) => parse_json_lines(raw).with_context(|| {
            format!("normal JSON parse failed ({json_error}); JSON lines parse also failed")
        }),
    }
}

fn unwrap_records(value: Value) -> Result<Vec<Value>> {
    match value {
        Value::Array(records) => Ok(records),
        Value::Object(mut object) => {
            for key in ["entries", "data", "requests", "test_data"] {
                if let Some(records) = object.remove(key) {
                    return match records {
                        Value::Array(records) => Ok(records),
                        _ => Err(anyhow!("top-level {key} field is not an array")),
                    };
                }
            }
            let keys = object.keys().cloned().collect::<Vec<_>>().join(", ");
            Err(anyhow!(
                "top-level object does not contain a supported record array field \
                 (expected one of entries, data, requests, test_data; keys: {keys})"
            ))
        }
        _ => Err(anyhow!("top-level JSON must be an array or object")),
    }
}

fn parse_json_lines(raw: &str) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    for (line_index, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_str(line)
            .with_context(|| format!("invalid JSON on line {}", line_index + 1))?;
        records.push(value);
    }
    Ok(records)
}

fn build_output(
    total: usize,
    cli: &Cli,
    engine: &SearchEngine,
    stats: &EvalStats,
    elapsed_ms: f64,
) -> EvalOutput {
    let matrix = stats.matrix;
    let evaluated = matrix.evaluated();
    let weighted_errors = matrix.weighted_errors();
    let denominator = evaluated as f64;

    EvalOutput {
        total,
        evaluated,
        true_positive_detections: matrix.true_positive_detections,
        true_negative_detections: matrix.true_negative_detections,
        false_positive_detections: matrix.false_positive_detections,
        false_negative_detections: matrix.false_negative_detections,
        weighted_errors_e: weighted_errors,
        weighted_errors_e_with_errors_as_fn: weighted_errors
            + 3 * (stats.parse_errors + stats.search_errors),
        failure_rate: ratio(weighted_errors, denominator),
        error_rate_epsilon: ratio(weighted_errors, denominator),
        accuracy: ratio(
            matrix.true_positive_detections + matrix.true_negative_detections,
            denominator,
        ),
        fraud_recall: ratio(
            matrix.true_positive_detections,
            (matrix.true_positive_detections + matrix.false_negative_detections) as f64,
        ),
        fraud_precision: ratio(
            matrix.true_positive_detections,
            (matrix.true_positive_detections + matrix.false_positive_detections) as f64,
        ),
        legit_recall: ratio(
            matrix.true_negative_detections,
            (matrix.true_negative_detections + matrix.false_positive_detections) as f64,
        ),
        elapsed_ms,
        avg_us_per_record: if evaluated == 0 {
            0.0
        } else {
            elapsed_ms * 1_000.0 / evaluated as f64
        },
        artifact_dir: cli.artifact_dir.display().to_string(),
        probe_count: engine.meta.probe_count,
        cluster_count: engine.meta.cluster_count,
        adaptive_probing_enabled: engine.adaptive_probing_enabled(),
        adaptive_initial_probes: engine.initial_probe_count(),
        parse_errors: stats.parse_errors,
        search_errors: stats.search_errors,
    }
}

fn ratio(numerator: u64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator as f64 / denominator
    }
}

fn env_path(name: &str, default: &str) -> PathBuf {
    env::var(name)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow!("{flag} requires a value"))
}

fn parse_usize(raw: String, flag: &str) -> Result<usize> {
    raw.parse()
        .with_context(|| format!("{flag} requires a non-negative integer"))
}

fn usage() -> &'static str {
    "usage: eval_ann --input <path> [--artifact-dir <path>] [--normalization <path>] \
     [--mcc-risk <path>] [--limit <n>] [--print-mismatches <n>] \
     [--expected-format auto|approved|label]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_expected_labels() {
        assert_eq!(
            extract_expected_approved(&json!({"expected": "fraud"}), ExpectedFormat::Auto),
            Some(false)
        );
        assert_eq!(
            extract_expected_approved(&json!({"label": "legit"}), ExpectedFormat::Auto),
            Some(true)
        );
        assert_eq!(
            extract_expected_approved(
                &json!({"expected": {"approved": false}}),
                ExpectedFormat::Auto
            ),
            Some(false)
        );
        assert_eq!(
            extract_expected_approved(&json!({"approved": true}), ExpectedFormat::Approved),
            Some(true)
        );
        assert_eq!(
            extract_expected_approved(&json!({"approved": true}), ExpectedFormat::Label),
            None
        );
    }

    #[test]
    fn confusion_matrix_update_matches_official_meaning() {
        let mut matrix = ConfusionMatrix::default();
        assert_eq!(matrix.update(false, false), DetectionKind::TruePositive);
        assert_eq!(matrix.update(true, true), DetectionKind::TrueNegative);
        assert_eq!(matrix.update(true, false), DetectionKind::FalsePositive);
        assert_eq!(matrix.update(false, true), DetectionKind::FalseNegative);

        assert_eq!(matrix.true_positive_detections, 1);
        assert_eq!(matrix.true_negative_detections, 1);
        assert_eq!(matrix.false_positive_detections, 1);
        assert_eq!(matrix.false_negative_detections, 1);
        assert_eq!(matrix.weighted_errors(), 4);
        assert_eq!(matrix.evaluated(), 4);
    }

    #[test]
    fn unwraps_supported_top_level_shapes() {
        assert_eq!(
            unwrap_records(json!([{"id": 1}, {"id": 2}])).unwrap().len(),
            2
        );
        let records = unwrap_records(json!({
            "references_checksum_sha256": "abc",
            "stats": { "total": 1 },
            "entries": [
                {
                    "expected_approved": true,
                    "expected_fraud_score": 0.0,
                    "request": {
                        "id": "tx-1",
                        "transaction": {
                            "amount": 100.0,
                            "installments": 1,
                            "requested_at": "2026-03-11T18:45:53Z"
                        },
                        "customer": {
                            "avg_amount": 50.0,
                            "tx_count_24h": 1,
                            "known_merchants": []
                        },
                        "merchant": {
                            "id": "MERC-001",
                            "mcc": "5411",
                            "avg_amount": 50.0
                        },
                        "terminal": {
                            "is_online": false,
                            "card_present": true,
                            "km_from_home": 10.0
                        },
                        "last_transaction": null
                    }
                }
            ]
        }))
        .unwrap();
        assert_eq!(records.len(), 1);
        let extracted = extract_case(&records[0], ExpectedFormat::Auto).unwrap();
        assert_eq!(extracted.expected_approved, true);
        assert_eq!(extracted.request.get("id"), Some(&json!("tx-1")));
        assert_eq!(
            unwrap_records(json!({"data": [{"id": 1}]})).unwrap().len(),
            1
        );
        assert_eq!(
            unwrap_records(json!({"requests": [{"id": 1}, {"id": 2}]}))
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            unwrap_records(json!({"test_data": [{"id": 1}]}))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn errors_with_object_keys_when_top_level_wrapper_is_missing() {
        let error = unwrap_records(json!({
            "references_checksum_sha256": "abc",
            "stats": { "total": 1 }
        }))
        .unwrap_err()
        .to_string();

        assert!(error.contains("references_checksum_sha256"));
        assert!(error.contains("stats"));
    }

    #[test]
    fn parses_json_lines_when_normal_json_fails() {
        let records = parse_records("{\"expected\":\"fraud\"}\n{\"expected\":\"legit\"}\n")
            .expect("json lines parse");
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn extracts_flat_request_without_expected_fields() {
        let extracted = extract_case(
            &json!({
                "id": "tx-1",
                "transaction": {},
                "customer": {},
                "merchant": {},
                "terminal": {},
                "last_transaction": null,
                "expected": "fraud"
            }),
            ExpectedFormat::Auto,
        )
        .unwrap();
        assert_eq!(extracted.expected_approved, false);
        assert!(extracted.request.get("expected").is_none());
        assert_eq!(extracted.request.get("id"), Some(&json!("tx-1")));
    }
}
