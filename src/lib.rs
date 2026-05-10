pub mod search;
pub mod simd;

use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use chrono::{Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};

pub const DIMENSIONS: usize = 14;
pub const PACKED_DIMENSIONS: usize = 16;
pub const TOP_K: usize = 5;
pub const ARTIFACT_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct FraudRequest {
    pub id: String,
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Transaction {
    pub amount: f32,
    pub installments: u32,
    pub requested_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FraudResponse {
    pub approved: bool,
    pub fraud_score: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceRecord {
    pub vector: [f32; DIMENSIONS],
    pub label: String,
}

pub type MccRiskMap = HashMap<String, f32>;

pub fn load_normalization(path: impl AsRef<Path>) -> Result<Normalization> {
    let raw = fs::read_to_string(path.as_ref())
        .with_context(|| format!("failed to read {}", path.as_ref().display()))?;
    serde_json::from_str(&raw).context("failed to parse normalization json")
}

pub fn load_mcc_risk(path: impl AsRef<Path>) -> Result<MccRiskMap> {
    let raw = fs::read_to_string(path.as_ref())
        .with_context(|| format!("failed to read {}", path.as_ref().display()))?;
    serde_json::from_str(&raw).context("failed to parse mcc risk json")
}

pub fn vectorize(
    request: &FraudRequest,
    normalization: &Normalization,
    mcc_risk: &MccRiskMap,
) -> Result<[f32; DIMENSIONS]> {
    let requested_at = chrono::DateTime::parse_from_rfc3339(&request.transaction.requested_at)
        .with_context(|| format!("invalid requested_at {}", request.transaction.requested_at))?
        .with_timezone(&Utc);

    let amount_vs_avg = if request.customer.avg_amount <= 0.0 {
        1.0
    } else {
        (request.transaction.amount / request.customer.avg_amount)
            / normalization.amount_vs_avg_ratio
    };

    let mut vector = [0.0; DIMENSIONS];
    vector[0] = clamp01(request.transaction.amount / normalization.max_amount);
    vector[1] = clamp01(request.transaction.installments as f32 / normalization.max_installments);
    vector[2] = clamp01(amount_vs_avg);
    vector[3] = requested_at.hour() as f32 / 23.0;
    vector[4] = requested_at.weekday().num_days_from_monday() as f32 / 6.0;

    match &request.last_transaction {
        Some(last_tx) => {
            let last_ts = chrono::DateTime::parse_from_rfc3339(&last_tx.timestamp)
                .with_context(|| {
                    format!("invalid last_transaction.timestamp {}", last_tx.timestamp)
                })?
                .with_timezone(&Utc);
            let minutes = requested_at
                .signed_duration_since(last_ts)
                .num_minutes()
                .max(0) as f32;
            vector[5] = clamp01(minutes / normalization.max_minutes);
            vector[6] = clamp01(last_tx.km_from_current / normalization.max_km);
        }
        None => {
            vector[5] = -1.0;
            vector[6] = -1.0;
        }
    }

    vector[7] = clamp01(request.terminal.km_from_home / normalization.max_km);
    vector[8] = clamp01(request.customer.tx_count_24h as f32 / normalization.max_tx_count_24h);
    vector[9] = if request.terminal.is_online { 1.0 } else { 0.0 };
    vector[10] = if request.terminal.card_present {
        1.0
    } else {
        0.0
    };
    vector[11] = if request
        .customer
        .known_merchants
        .iter()
        .any(|known| known == &request.merchant.id)
    {
        0.0
    } else {
        1.0
    };
    vector[12] = *mcc_risk.get(&request.merchant.mcc).unwrap_or(&0.5);
    vector[13] = clamp01(request.merchant.avg_amount / normalization.max_merchant_avg_amount);

    Ok(vector)
}

pub fn quantize_vector(vector: &[f32; DIMENSIONS]) -> [i8; DIMENSIONS] {
    let mut out = [0_i8; DIMENSIONS];
    for (idx, value) in vector.iter().enumerate() {
        let scaled = value.clamp(-1.0, 1.0) * 127.0;
        out[idx] = scaled.round() as i8;
    }
    out
}

pub fn quantize_vector_padded(vector: &[f32; DIMENSIONS]) -> [i8; PACKED_DIMENSIONS] {
    let quantized = quantize_vector(vector);
    let mut out = [0_i8; PACKED_DIMENSIONS];
    out[..DIMENSIONS].copy_from_slice(&quantized);
    out
}

pub fn pad_centroid(centroid: &[f32; DIMENSIONS]) -> [f32; PACKED_DIMENSIONS] {
    let mut out = [0.0_f32; PACKED_DIMENSIONS];
    out[..DIMENSIONS].copy_from_slice(centroid);
    out
}

pub fn dequantize_component(value: i8) -> f32 {
    value as f32 / 127.0
}

pub fn squared_distance_i8_scalar(query: &[i8; PACKED_DIMENSIONS], candidate: &[u8]) -> u32 {
    let mut sum = 0_u32;
    for idx in 0..PACKED_DIMENSIONS {
        let delta = query[idx] as i32 - candidate[idx] as i8 as i32;
        sum += (delta * delta) as u32;
    }
    sum
}

pub fn squared_distance_f32_scalar(
    query: &[f32; PACKED_DIMENSIONS],
    candidate: &[f32; PACKED_DIMENSIONS],
) -> f32 {
    let mut sum = 0.0_f32;
    for idx in 0..PACKED_DIMENSIONS {
        let delta = query[idx] - candidate[idx];
        sum += delta * delta;
    }
    sum
}

pub fn score_neighbors(labels: &[u8]) -> FraudResponse {
    let fraud_count = labels.iter().filter(|label| **label == 1).count();
    let fraud_score = fraud_count as f32 / TOP_K as f32;
    FraudResponse {
        approved: fraud_score < 0.6,
        fraud_score,
    }
}

pub fn deny_response() -> FraudResponse {
    FraudResponse {
        approved: false,
        fraud_score: 1.0,
    }
}

pub fn heuristic_response(
    request: &FraudRequest,
    normalization: &Normalization,
    mcc_risk: &MccRiskMap,
) -> FraudResponse {
    let mut score = 0.0_f32;
    let avg_amount = if request.customer.avg_amount <= 0.0 {
        normalization.max_amount
    } else {
        request.customer.avg_amount
    };

    if request.transaction.amount > avg_amount * 4.0 {
        score += 0.25;
    }
    if !request
        .customer
        .known_merchants
        .iter()
        .any(|known| known == &request.merchant.id)
    {
        score += 0.2;
    }
    if request.terminal.is_online {
        score += 0.15;
    }
    if !request.terminal.card_present {
        score += 0.15;
    }
    if request.terminal.km_from_home > normalization.max_km * 0.4 {
        score += 0.15;
    }
    score += mcc_risk.get(&request.merchant.mcc).copied().unwrap_or(0.5) * 0.1;

    let fraud_score = score.clamp(0.0, 1.0);
    FraudResponse {
        approved: fraud_score < 0.6,
        fraud_score: ((fraud_score * TOP_K as f32).round() / TOP_K as f32).clamp(0.0, 1.0),
    }
}

pub fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

pub fn validate_reference_record(record: &ReferenceRecord) -> Result<()> {
    if record.label != "fraud" && record.label != "legit" {
        return Err(anyhow!("unexpected label {}", record.label));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f32, right: f32) {
        assert!((left - right).abs() < 0.001, "left={left} right={right}");
    }

    fn test_normalization() -> Normalization {
        Normalization {
            max_amount: 10000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10000.0,
        }
    }

    fn test_mcc() -> MccRiskMap {
        HashMap::from([("5411".to_string(), 0.15), ("7802".to_string(), 0.75)])
    }

    #[test]
    fn vectorizes_legit_example() {
        let request: FraudRequest = serde_json::from_str(
            r#"{
                "id": "tx-1329056812",
                "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
                "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
                "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
                "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
                "last_transaction": null
            }"#,
        )
        .unwrap();

        let vector = vectorize(&request, &test_normalization(), &test_mcc()).unwrap();

        let expected = [
            0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1.0, -1.0, 0.0292, 0.15, 0.0, 1.0, 0.0, 0.15,
            0.006,
        ];
        for (left, right) in vector.iter().zip(expected.iter()) {
            approx_eq(*left, *right);
        }
    }

    #[test]
    fn vectorizes_fraud_example() {
        let request: FraudRequest = serde_json::from_str(
            r#"{
                "id": "tx-3330991687",
                "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
                "customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
                "merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
                "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
                "last_transaction": null
            }"#,
        )
        .unwrap();

        let vector = vectorize(&request, &test_normalization(), &test_mcc()).unwrap();

        let expected = [
            0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1.0, -1.0, 0.9523, 1.0, 0.0, 1.0, 1.0, 0.75,
            0.0055,
        ];
        for (left, right) in vector.iter().zip(expected.iter()) {
            approx_eq(*left, *right);
        }
    }

    #[test]
    fn quantization_preserves_sentinel() {
        let vector = [
            0.0, 1.0, 0.5, 0.25, 0.75, -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.25,
        ];
        let quantized = quantize_vector(&vector);
        assert_eq!(quantized[5], -127);
        assert_eq!(quantized[6], -127);
    }

    #[test]
    fn padded_quantization_zeroes_extra_lanes() {
        let vector = [
            0.0, 1.0, 0.5, 0.25, 0.75, -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.25,
        ];
        let quantized = quantize_vector_padded(&vector);
        assert_eq!(quantized[5], -127);
        assert_eq!(quantized[6], -127);
        assert_eq!(quantized[14], 0);
        assert_eq!(quantized[15], 0);
    }

    #[test]
    fn scores_top_five_neighbors() {
        let response = score_neighbors(&[1, 1, 1, 0, 0]);
        assert_eq!(
            response,
            FraudResponse {
                approved: false,
                fraud_score: 0.6
            }
        );
    }
}
