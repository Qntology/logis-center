//! Navigable Small World (NSW) approximate nearest-neighbor index.
//!
//! Implements a single-layer NSW graph — a simplified HNSW variant that is
//! fast enough for caches up to ~100k entries while keeping the implementation
//! self-contained and free of external dependencies.
//!
//! # Algorithm sketch
//!
//! - **Insert**: greedily traverse the graph from a random (deterministic)
//!   entry point, collecting the `ef_construct` nearest nodes.  Connect the
//!   new node to at most `max_connections` of them.  Prune neighbours that
//!   exceed `max_connections`.
//! - **Search**: repeat the greedy traversal, expanding `ef_search` candidates,
//!   and return the top-k by cosine similarity.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_runtime::embedding_index::{EmbeddingIndex, NswConfig};
//!
//! let mut index: EmbeddingIndex<&str> = EmbeddingIndex::new(4);
//! let id = index.insert(vec![1.0, 0.0, 0.0, 0.0], "doc-a");
//! let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1);
//! assert_eq!(results[0].1, &"doc-a");
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length unit vectors.
///
/// Both inputs are assumed to already be L2-normalised.  Returns a value in
/// `[-1.0, 1.0]`; returns `0.0` for empty or mismatched inputs.
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

/// L2-normalise `v` in place.  Leaves zero-vectors unchanged.
#[inline]
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NswNode (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// A single node stored in the NSW graph.
struct NswNode {
    /// Unique numeric identifier (equals the node's position in `NswIndex::nodes`).
    id: usize,
    /// L2-normalised embedding vector.
    vector: Vec<f32>,
    /// Indices of connected neighbours in `NswIndex::nodes`.
    neighbors: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// NswConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the NSW approximate nearest-neighbor graph.
#[derive(Debug, Clone)]
pub struct NswConfig {
    /// Maximum number of bidirectional connections per node during construction
    /// (default: 16).  Higher values improve recall at the cost of memory and
    /// insertion time.
    pub max_connections: usize,
    /// Number of candidates to explore during search (default: 64).  Higher
    /// values improve recall at the cost of query latency.
    pub ef_search: usize,
    /// Number of candidates to explore during insertion (default: 32).  Higher
    /// values improve graph quality at the cost of insertion latency.
    pub ef_construct: usize,
}

impl Default for NswConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_search: 64,
            ef_construct: 32,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NswSearchResult
// ─────────────────────────────────────────────────────────────────────────────

/// A single result from an NSW nearest-neighbor search.
#[derive(Debug, Clone)]
pub struct NswSearchResult {
    /// The node's unique identifier (stable across insertions).
    pub id: usize,
    /// Cosine similarity score between the query and this node's vector.
    pub score: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// NswIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Navigable Small World graph index for approximate nearest-neighbor search.
///
/// This is a single-layer NSW — the multi-layer hierarchical variant (HNSW) is
/// outside scope.  Performance is excellent for corpora up to ~100k entries.
pub struct NswIndex {
    nodes: Vec<NswNode>,
    config: NswConfig,
    dim: usize,
    /// Simple deterministic counter used instead of a random entry point so
    /// that behaviour is reproducible without the `rand` crate.
    entry_counter: usize,
}

impl NswIndex {
    /// Create an empty NSW index for `dim`-dimensional vectors.
    pub fn new(dim: usize, config: NswConfig) -> Self {
        Self {
            nodes: Vec::new(),
            config,
            dim,
            entry_counter: 0,
        }
    }

    // ── Insertion ─────────────────────────────────────────────────────────────

    /// Insert a normalised copy of `vector` with the given `id`.
    ///
    /// 1. Finds `ef_construct` nearest existing nodes via greedy search.
    /// 2. Connects the new node to at most `max_connections` of them.
    /// 3. Prunes the neighbours' connection lists if they exceed `max_connections`.
    ///
    /// Complexity: O(M × ef_construct) amortised where M = `max_connections`.
    pub fn insert(&mut self, id: usize, vector: Vec<f32>) {
        let mut v = vector;
        // Pad or truncate to match declared dimensionality.
        v.resize(self.dim, 0.0);
        l2_normalize(&mut v);

        let new_idx = self.nodes.len();

        if self.nodes.is_empty() {
            // First node — no edges to add yet.
            self.nodes.push(NswNode {
                id,
                vector: v,
                neighbors: Vec::new(),
            });
            self.entry_counter = 0;
            return;
        }

        // Pick a deterministic entry point by rotating through existing nodes.
        let entry = self.entry_counter % self.nodes.len();
        self.entry_counter += 1;

        // Find ef_construct nearest candidates.
        let ef = self.config.ef_construct;
        let candidates = self.greedy_search(&v, entry, ef);

        // Keep at most max_connections neighbours.
        let max_conn = self.config.max_connections;
        let neighbor_indices: Vec<usize> = candidates
            .iter()
            .take(max_conn)
            .map(|(node_idx, _)| *node_idx)
            .collect();

        // Add the new node.
        self.nodes.push(NswNode {
            id,
            vector: v.clone(),
            neighbors: neighbor_indices.clone(),
        });

        // Add back-edges and prune if needed.
        for &nb_idx in &neighbor_indices {
            self.nodes[nb_idx].neighbors.push(new_idx);
            if self.nodes[nb_idx].neighbors.len() > max_conn {
                self.prune_neighbors(nb_idx, max_conn);
            }
        }
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Return the top-`top_k` approximate nearest neighbors of `query`.
    ///
    /// Uses a greedy graph traversal starting from a deterministic entry point,
    /// expanding at most `ef_search` candidates.  Results are sorted by cosine
    /// similarity in descending order.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<NswSearchResult> {
        if self.nodes.is_empty() || top_k == 0 {
            return Vec::new();
        }

        // Normalise the query locally.
        let mut q = query.to_vec();
        q.resize(self.dim, 0.0);
        l2_normalize(&mut q);

        // Use node 0 as a stable entry point for search (read-only, no mutation).
        let entry = 0;
        let ef = self.config.ef_search;
        let mut candidates = self.greedy_search(&q, entry, ef);

        // Sort descending by score.
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        candidates
            .into_iter()
            .map(|(node_idx, score)| NswSearchResult {
                id: self.nodes[node_idx].id,
                score,
            })
            .collect()
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of vectors stored in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The embedding dimensionality this index was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Greedy beam search from `entry` node, returning up to `ef` candidates
    /// as `(node_index, cosine_similarity)` pairs.
    ///
    /// The implementation maintains two sets:
    /// - `visited`: bit-set of already-explored node indices.
    /// - `candidates`: max-heap of (score, node_idx) to explore next.
    /// - `results`: the ef best nodes seen so far.
    fn greedy_search(&self, query: &[f32], entry: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        use std::cmp::Ordering;
        use std::collections::{BinaryHeap, HashSet};

        /// Wrapper to allow f32 in BinaryHeap (max-heap by score).
        #[derive(PartialEq)]
        struct Scored(f32, usize);

        impl Eq for Scored {}

        impl PartialOrd for Scored {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for Scored {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0
                    .partial_cmp(&other.0)
                    .unwrap_or(Ordering::Equal)
                    .then(self.1.cmp(&other.1))
            }
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let entry_score = cosine_sim(query, &self.nodes[entry].vector);
        visited.insert(entry);

        // `frontier` is a max-heap of nodes to expand (best first).
        let mut frontier: BinaryHeap<Scored> = BinaryHeap::new();
        frontier.push(Scored(entry_score, entry));

        // `results` keeps the best `ef` nodes found so far.
        let mut results: Vec<(usize, f32)> = vec![(entry, entry_score)];

        while let Some(Scored(_, node_idx)) = frontier.pop() {
            // If results already has ef entries and the worst result in results
            // is better than anything remaining in the frontier, we can stop.
            if results.len() >= ef {
                let worst_result = results
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::INFINITY, f32::min);
                // All remaining frontier nodes are at most as good as `node_idx`
                // (max-heap), so check against the worst we currently keep.
                let node_score = results
                    .iter()
                    .find(|(i, _)| *i == node_idx)
                    .map(|(_, s)| *s)
                    .unwrap_or(f32::NEG_INFINITY);
                if node_score < worst_result && frontier.is_empty() {
                    break;
                }
            }

            // Expand neighbours.
            for &nb_idx in &self.nodes[node_idx].neighbors {
                if visited.contains(&nb_idx) {
                    continue;
                }
                visited.insert(nb_idx);

                let nb_score = cosine_sim(query, &self.nodes[nb_idx].vector);
                frontier.push(Scored(nb_score, nb_idx));
                results.push((nb_idx, nb_score));

                // Keep results bounded at ef (drop worst).
                if results.len() > ef {
                    let worst_idx = results
                        .iter()
                        .enumerate()
                        .min_by(|a, b| {
                            a.1 .1
                                .partial_cmp(&b.1 .1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .expect("results is non-empty");
                    results.swap_remove(worst_idx);
                }
            }
        }

        results
    }

    /// Prune the neighbor list of node at `node_idx` to at most `max_conn`
    /// connections, keeping the `max_conn` closest by cosine similarity.
    fn prune_neighbors(&mut self, node_idx: usize, max_conn: usize) {
        let v = self.nodes[node_idx].vector.clone();
        let neighbors = &self.nodes[node_idx].neighbors;

        // Score each current neighbour.
        let mut scored: Vec<(usize, f32)> = neighbors
            .iter()
            .map(|&nb| {
                let score = cosine_sim(&v, &self.nodes[nb].vector);
                (nb, score)
            })
            .collect();

        // Keep highest-scoring connections.
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_conn);

        self.nodes[node_idx].neighbors = scored.into_iter().map(|(nb, _)| nb).collect();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingIndex<T>
// ─────────────────────────────────────────────────────────────────────────────

/// Combined NSW graph index with per-entry metadata storage.
///
/// `T` is any cloneable metadata type — e.g. a `String` payload, a struct, or
/// a raw identifier.
///
/// ```rust
/// use oxibonsai_runtime::embedding_index::EmbeddingIndex;
///
/// let mut idx: EmbeddingIndex<String> = EmbeddingIndex::new(3);
/// idx.insert(vec![1.0, 0.0, 0.0], "vec-a".to_string());
/// idx.insert(vec![0.0, 1.0, 0.0], "vec-b".to_string());
///
/// let results = idx.search(&[1.0, 0.0, 0.0], 1);
/// assert_eq!(results[0].1, &"vec-a".to_string());
/// ```
pub struct EmbeddingIndex<T: Clone> {
    graph: NswIndex,
    /// Parallel metadata store: `metadata[i] = (id, metadata_value)`.
    metadata: Vec<(usize, T)>,
    next_id: usize,
}

impl<T: Clone> EmbeddingIndex<T> {
    /// Create a new index for `dim`-dimensional vectors with default NSW config.
    pub fn new(dim: usize) -> Self {
        Self::new_with_config(dim, NswConfig::default())
    }

    /// Create a new index with a custom [`NswConfig`].
    pub fn new_with_config(dim: usize, config: NswConfig) -> Self {
        Self {
            graph: NswIndex::new(dim, config),
            metadata: Vec::new(),
            next_id: 0,
        }
    }

    /// Insert a vector with associated metadata.
    ///
    /// Returns the stable numeric ID assigned to this entry.
    pub fn insert(&mut self, vector: Vec<f32>, meta: T) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.graph.insert(id, vector);
        self.metadata.push((id, meta));
        id
    }

    /// Search for the top-`top_k` nearest neighbors of `query`.
    ///
    /// Returns a `Vec` of `(NswSearchResult, &T)` pairs sorted by descending
    /// cosine similarity.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(NswSearchResult, &T)> {
        let results = self.graph.search(query, top_k);
        results
            .into_iter()
            .filter_map(|r| {
                // Look up metadata by id.
                self.metadata
                    .iter()
                    .find(|(id, _)| *id == r.id)
                    .map(|(_, meta)| (r, meta))
            })
            .collect()
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Returns `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(values: &[f32]) -> Vec<f32> {
        let mut v = values.to_vec();
        l2_normalize(&mut v);
        v
    }

    // ── NswIndex ──────────────────────────────────────────────────────────────

    #[test]
    fn test_nsw_index_empty() {
        let idx = NswIndex::new(4, NswConfig::default());
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.dim(), 4);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_nsw_index_single_insert() {
        let mut idx = NswIndex::new(4, NswConfig::default());
        idx.insert(0, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!(
            (results[0].score - 1.0).abs() < 1e-5,
            "score={}",
            results[0].score
        );
    }

    #[test]
    fn test_nsw_index_search_exact() {
        let mut idx = NswIndex::new(3, NswConfig::default());
        let v = unit_vec(&[1.0, 2.0, 3.0]);
        idx.insert(42, v.clone());
        let results = idx.search(&v, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
        assert!(
            (results[0].score - 1.0).abs() < 1e-5,
            "score={}",
            results[0].score
        );
    }

    #[test]
    fn test_nsw_index_search_nearest() {
        let mut idx = NswIndex::new(2, NswConfig::default());
        // Insert three vectors; query is closest to id=1.
        idx.insert(0, unit_vec(&[1.0, 0.0])); // along x-axis
        idx.insert(1, unit_vec(&[0.0, 1.0])); // along y-axis
        idx.insert(2, unit_vec(&[-1.0, 0.0])); // negative x-axis

        let query = unit_vec(&[0.1, 0.9]); // close to y-axis
        let results = idx.search(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].id, 1,
            "nearest should be y-axis vector, got id={}",
            results[0].id
        );
    }

    #[test]
    fn test_nsw_index_many_vectors() {
        let dim = 8;
        let config = NswConfig {
            max_connections: 8,
            ef_search: 32,
            ef_construct: 16,
        };
        let mut idx = NswIndex::new(dim, config);

        // Insert 100 random-ish deterministic vectors.
        for i in 0..100usize {
            let mut v: Vec<f32> = (0..dim)
                .map(|d| {
                    // deterministic pseudo-random using wrapping arithmetic
                    let x = (i as u64)
                        .wrapping_mul(6364136223846793005u64)
                        .wrapping_add((d as u64).wrapping_mul(1442695040888963407u64));
                    let x = x ^ (x >> 33);
                    let x = x.wrapping_mul(0xff51afd7ed558ccdu64);
                    let x = x ^ (x >> 33);
                    (x as i64) as f32 / i64::MAX as f32
                })
                .collect();
            l2_normalize(&mut v);
            idx.insert(i, v);
        }

        assert_eq!(idx.len(), 100);

        // A known query: a unit vector along the first dimension.
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let results = idx.search(&query, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Scores should be in descending order.
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score - 1e-5,
                "scores not sorted: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    // ── EmbeddingIndex ────────────────────────────────────────────────────────

    #[test]
    fn test_embedding_index_insert_and_search() {
        let mut idx: EmbeddingIndex<u32> = EmbeddingIndex::new(4);
        idx.insert(unit_vec(&[1.0, 0.0, 0.0, 0.0]), 100);
        idx.insert(unit_vec(&[0.0, 1.0, 0.0, 0.0]), 200);
        idx.insert(unit_vec(&[0.0, 0.0, 1.0, 0.0]), 300);

        let results = idx.search(&unit_vec(&[1.0, 0.0, 0.0, 0.0]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, 100u32);
    }

    #[test]
    fn test_embedding_index_metadata_returned() {
        let mut idx: EmbeddingIndex<String> = EmbeddingIndex::new(3);
        let id = idx.insert(unit_vec(&[1.0, 1.0, 0.0]), "hello world".to_string());
        assert_eq!(id, 0);
        let results = idx.search(&unit_vec(&[1.0, 1.0, 0.0]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, &"hello world".to_string());
        assert!((results[0].0.score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_nsw_config_defaults() {
        let cfg = NswConfig::default();
        assert_eq!(cfg.max_connections, 16);
        assert_eq!(cfg.ef_search, 64);
        assert_eq!(cfg.ef_construct, 32);
    }
}
