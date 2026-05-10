use std::{
    cmp::Ordering,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::{
    DIMENSIONS, FraudResponse, TOP_K, dequantize_component, quantize_vector, score_neighbors,
    squared_distance_i8,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMeta {
    pub version: u32,
    pub dimensions: usize,
    pub vector_count: u64,
    pub cluster_count: usize,
    pub probe_count: usize,
    pub list_offsets: Vec<u64>,
    pub list_lengths: Vec<u32>,
}

pub struct SearchEngine {
    pub meta: ArtifactMeta,
    centroids: Vec<[f32; DIMENSIONS]>,
    vectors: Mmap,
    labels: Mmap,
    _artifact_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct Neighbor {
    distance: u32,
    label: u8,
}

struct TopK {
    entries: [Neighbor; TOP_K],
    len: usize,
}

impl TopK {
    fn new() -> Self {
        Self {
            entries: [Neighbor {
                distance: u32::MAX,
                label: 0,
            }; TOP_K],
            len: 0,
        }
    }

    fn insert(&mut self, distance: u32, label: u8) {
        if self.len < TOP_K {
            self.entries[self.len] = Neighbor { distance, label };
            self.len += 1;
            return;
        }

        let mut worst_idx = 0;
        for idx in 1..TOP_K {
            if self.entries[idx].distance > self.entries[worst_idx].distance {
                worst_idx = idx;
            }
        }

        if distance < self.entries[worst_idx].distance {
            self.entries[worst_idx] = Neighbor { distance, label };
        }
    }

    fn labels(&self) -> [u8; TOP_K] {
        let mut labels = [0_u8; TOP_K];
        for (dst, entry) in labels.iter_mut().zip(self.entries.iter()) {
            *dst = entry.label;
        }
        labels
    }
}

impl SearchEngine {
    pub fn open(path: impl AsRef<Path>, probe_override: Option<usize>) -> Result<Self> {
        let dir = path.as_ref().to_path_buf();
        let meta_raw = std::fs::read_to_string(dir.join("meta.json"))
            .with_context(|| format!("failed to read {}", dir.join("meta.json").display()))?;
        let mut meta: ArtifactMeta =
            serde_json::from_str(&meta_raw).context("invalid meta.json")?;
        if let Some(probe_count) = probe_override {
            meta.probe_count = probe_count.max(1).min(meta.cluster_count);
        }

        if meta.dimensions != DIMENSIONS {
            return Err(anyhow!(
                "artifact dimension mismatch: expected {} got {}",
                DIMENSIONS,
                meta.dimensions
            ));
        }

        let centroids = load_centroids(&dir.join("centroids.bin"), meta.cluster_count)?;
        let vectors_file = File::open(dir.join("vectors.bin"))
            .with_context(|| format!("failed to open {}", dir.join("vectors.bin").display()))?;
        let labels_file = File::open(dir.join("labels.bin"))
            .with_context(|| format!("failed to open {}", dir.join("labels.bin").display()))?;

        let vectors = unsafe { Mmap::map(&vectors_file) }.context("failed to mmap vectors.bin")?;
        let labels = unsafe { Mmap::map(&labels_file) }.context("failed to mmap labels.bin")?;

        Ok(Self {
            meta,
            centroids,
            vectors,
            labels,
            _artifact_dir: dir,
        })
    }

    pub fn score(&self, query: &[f32; DIMENSIONS]) -> Result<FraudResponse> {
        let quantized = quantize_vector(query);
        let mut ranked_clusters: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| (centroid_distance(query, centroid), idx))
            .collect();
        ranked_clusters
            .sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));

        let mut top_k = TopK::new();
        let mut scanned = 0_usize;
        for (_, cluster_idx) in ranked_clusters {
            self.scan_cluster(cluster_idx, &quantized, &mut top_k)?;
            scanned += 1;
            if scanned >= self.meta.probe_count && top_k.len >= TOP_K {
                break;
            }
        }

        if top_k.len < TOP_K {
            return Err(anyhow!("not enough candidates to score top-5 neighbors"));
        }

        Ok(score_neighbors(&top_k.labels()))
    }

    fn scan_cluster(
        &self,
        cluster_idx: usize,
        query: &[i8; DIMENSIONS],
        top_k: &mut TopK,
    ) -> Result<()> {
        let start = self.meta.list_offsets[cluster_idx] as usize;
        let len = self.meta.list_lengths[cluster_idx] as usize;
        for item_idx in 0..len {
            let absolute_idx = start + item_idx;
            let base = absolute_idx * DIMENSIONS;
            let candidate = self
                .vectors
                .get(base..base + DIMENSIONS)
                .ok_or_else(|| anyhow!("vector slice out of bounds"))?;
            let label = *self
                .labels
                .get(absolute_idx)
                .ok_or_else(|| anyhow!("label index out of bounds"))?;
            let distance = squared_distance_i8(query, candidate);
            top_k.insert(distance, label);
        }
        Ok(())
    }
}

fn load_centroids(path: &Path, cluster_count: usize) -> Result<Vec<[f32; DIMENSIONS]>> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut raw = Vec::new();
    file.read_to_end(&mut raw)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let expected_len = cluster_count * DIMENSIONS * std::mem::size_of::<f32>();
    if raw.len() != expected_len {
        return Err(anyhow!(
            "centroids.bin size mismatch: expected {expected_len} got {}",
            raw.len()
        ));
    }

    let mut centroids = Vec::with_capacity(cluster_count);
    for chunk in raw.chunks_exact(DIMENSIONS * 4) {
        let mut centroid = [0.0_f32; DIMENSIONS];
        for (idx, bytes) in chunk.chunks_exact(4).enumerate() {
            centroid[idx] = f32::from_le_bytes(bytes.try_into().unwrap());
        }
        centroids.push(centroid);
    }
    Ok(centroids)
}

fn centroid_distance(query: &[f32; DIMENSIONS], centroid: &[f32; DIMENSIONS]) -> f32 {
    let mut sum = 0.0_f32;
    for idx in 0..DIMENSIONS {
        let delta = query[idx] - centroid[idx];
        sum += delta * delta;
    }
    sum
}

pub fn centroid_from_quantized(chunk: &[i8]) -> [f32; DIMENSIONS] {
    let mut centroid = [0.0_f32; DIMENSIONS];
    for idx in 0..DIMENSIONS {
        centroid[idx] = dequantize_component(chunk[idx]);
    }
    centroid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_prefers_smallest_distance() {
        let mut top_k = TopK::new();
        top_k.insert(80, 0);
        top_k.insert(50, 1);
        top_k.insert(40, 1);
        top_k.insert(30, 0);
        top_k.insert(20, 1);
        top_k.insert(10, 1);

        let labels = top_k.labels();
        assert_eq!(labels.iter().filter(|label| **label == 1).count(), 4);
    }
}
