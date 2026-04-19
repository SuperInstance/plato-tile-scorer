//! plato-tile-scorer — Unified Tile Scoring Engine
//!
//! Fuses all fleet signals into one score:
//! - Temporal decay (freshness)
//! - Ghost score (usage fade)
//! - Belief (confidence × trust × relevance)
//! - Domain relevance (query↔tile match)
//! - Use frequency (popularity)

/// Per-tile signal scores (each 0.0–1.0).
#[derive(Debug, Clone, Default)]
pub struct TileSignal {
    pub temporal_score: f64,
    pub ghost_score: f64,
    pub belief_confidence: f64,
    pub belief_trust: f64,
    pub belief_relevance: f64,
    pub domain_relevance: f64,
    pub use_frequency: f64,
}

impl TileSignal {
    pub fn belief_composite(&self) -> f64 {
        let product: f64 = self.belief_confidence * self.belief_trust * self.belief_relevance;
        product.powf(1.0_f64 / 3.0_f64)
    }
}

/// Scoring weights.
#[derive(Debug, Clone)]
pub struct ScoreConfig {
    pub temporal_weight: f64,
    pub ghost_weight: f64,
    pub belief_weight: f64,
    pub domain_weight: f64,
    pub frequency_weight: f64,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        Self { temporal_weight: 0.25, ghost_weight: 0.15, belief_weight: 0.30, domain_weight: 0.20, frequency_weight: 0.10 }
    }
}

impl ScoreConfig {
    pub fn total_weight(&self) -> f64 {
        self.temporal_weight + self.ghost_weight + self.belief_weight + self.domain_weight + self.frequency_weight
    }
    pub fn normalize(&mut self) {
        let total = self.total_weight();
        if total > 0.0 { for w in [&mut self.temporal_weight, &mut self.ghost_weight, &mut self.belief_weight, &mut self.domain_weight, &mut self.frequency_weight] { *w /= total; } }
    }
}

/// A scored tile.
#[derive(Debug, Clone)]
pub struct ScoredTile { pub id: String, pub score: f64, pub signals: TileSignal }

/// Top-K selection.
#[derive(Debug, Clone, Default)]
pub struct TopK { pub items: Vec<ScoredTile>, pub k: usize }

impl TopK {
    pub fn new(k: usize) -> Self { Self { items: Vec::new(), k: k.max(1) } }
    pub fn try_insert(&mut self, tile: ScoredTile) {
        if self.items.len() < self.k {
            self.items.push(tile);
        } else if self.items.last().map_or(false, |last| tile.score > last.score) {
            self.items[self.k - 1] = tile;
        } else { return; }
        self.items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    }
    pub fn results(&self) -> &[ScoredTile] { &self.items }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
}

/// Unified tile scorer.
pub struct TileScorer { config: ScoreConfig }

impl Default for TileScorer { fn default() -> Self { Self::new() } }

impl TileScorer {
    pub fn new() -> Self { Self { config: ScoreConfig::default() } }
    pub fn with_config(config: ScoreConfig) -> Self { Self { config } }

    pub fn score(&self, signal: &TileSignal) -> f64 {
        let ghost_alive = 1.0 - signal.ghost_score;
        let belief = signal.belief_composite();
        self.config.temporal_weight * signal.temporal_score
            + self.config.ghost_weight * ghost_alive
            + self.config.belief_weight * belief
            + self.config.domain_weight * signal.domain_relevance
            + self.config.frequency_weight * signal.use_frequency
    }

    pub fn score_batch(&self, tiles: &[(String, TileSignal)], k: usize) -> TopK {
        let mut top = TopK::new(k);
        for (id, signal) in tiles {
            top.try_insert(ScoredTile { id: id.clone(), score: self.score(signal), signals: signal.clone() });
        }
        top
    }
}

// ── Decay Functions ──

pub fn temporal_decay(age: f64, validity_window: f64) -> f64 {
    if age <= 0.0 { return 1.0; }
    if validity_window <= 0.0 { return 0.0; }
    (1.0 - (age / validity_window)).max(0.0)
}

pub fn ghost_decay(current_ghost: f64, ticks_inactive: u64, rate: f64) -> f64 {
    (current_ghost + ticks_inactive as f64 * rate).min(1.0)
}

pub fn belief_decay(current: f64, rate: f64) -> f64 {
    (current + (0.5 - current) * rate).max(0.0).min(1.0)
}

pub fn frequency_normalize(use_count: u32, max_seen: u32) -> f64 {
    if max_seen == 0 { return 0.0; }
    (use_count as f64 / max_seen as f64).min(1.0)
}

pub fn domain_relevance(tile_domain: &str, query_domain: &str) -> f64 {
    if tile_domain.is_empty() || query_domain.is_empty() { return 0.0; }
    if tile_domain == query_domain { return 1.0; }
    let tw: Vec<&str> = tile_domain.split_whitespace().collect();
    let qw: Vec<&str> = query_domain.split_whitespace().collect();
    let inter = tw.iter().filter(|w| qw.iter().any(|q| q.eq_ignore_ascii_case(w))).count();
    let union = tw.len() + qw.len() - inter;
    if union == 0 { return 0.0; }
    inter as f64 / union as f64
}

pub fn safe_score(value: f64, fallback: f64) -> f64 {
    if value.is_nan() || value.is_infinite() { fallback } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> TileSignal {
        TileSignal { temporal_score: 1.0, ghost_score: 0.0, belief_confidence: 0.9, belief_trust: 0.8, belief_relevance: 0.7, domain_relevance: 1.0, use_frequency: 0.5 }
    }

    #[test] fn test_score_fresh() { assert!(TileScorer::new().score(&fresh()) > 0.7); }
    #[test] fn test_score_stale() {
        let s = TileSignal { temporal_score: 0.1, ghost_score: 0.9, belief_confidence: 0.3, belief_trust: 0.3, belief_relevance: 0.3, domain_relevance: 0.1, use_frequency: 0.0 };
        assert!(TileScorer::new().score(&s) < 0.3);
    }
    #[test] fn test_belief_composite() { let s = fresh(); assert!((s.belief_composite() - (0.9_f64*0.8*0.7).powf(1.0/3.0)).abs() < 0.001); }
    #[test] fn test_belief_zero() { assert!((TileSignal { belief_confidence: 0.0, belief_trust: 0.5, belief_relevance: 0.5, ..Default::default() }.belief_composite()).abs() < 0.001); }
    #[test] fn test_top_k() {
        let tiles = vec![("low".into(), TileSignal { temporal_score: 0.1, ..Default::default() }), ("high".into(), fresh())];
        let top = TileScorer::new().score_batch(&tiles, 1);
        assert_eq!(top.results()[0].id, "high");
    }
    #[test] fn test_top_k_empty() { assert!(TileScorer::new().score_batch(&[], 5).is_empty()); }
    #[test] fn test_temporal_fresh() { assert!((temporal_decay(0.0, 100.0) - 1.0).abs() < 0.001); }
    #[test] fn test_temporal_half() { assert!((temporal_decay(50.0, 100.0) - 0.5).abs() < 0.001); }
    #[test] fn test_temporal_expired() { assert!((temporal_decay(200.0, 100.0)).abs() < 0.001); }
    #[test] fn test_ghost_decay() { assert!((ghost_decay(0.0, 10, 0.05) - 0.5).abs() < 0.001); }
    #[test] fn test_ghost_capped() { assert!(ghost_decay(0.5, 1000, 0.1) <= 1.0); }
    #[test] fn test_belief_decay() { let b = belief_decay(1.0, 0.5); assert!(b < 1.0 && b > 0.5); }
    #[test] fn test_freq_norm() { assert!((frequency_normalize(50, 100) - 0.5).abs() < 0.001); }
    #[test] fn test_freq_zero_max() { assert!((frequency_normalize(50, 0)).abs() < 0.001); }
    #[test] fn test_domain_exact() { assert!((domain_relevance("math", "math") - 1.0).abs() < 0.001); }
    #[test] fn test_domain_partial() { let r = domain_relevance("math geometry", "math algebra"); assert!(r > 0.0 && r < 1.0); }
    #[test] fn test_domain_empty() { assert!((domain_relevance("", "math")).abs() < 0.001); }
    #[test] fn test_safe_nan() { assert!((safe_score(f64::NAN, 0.5) - 0.5).abs() < 0.001); }
    #[test] fn test_safe_inf() { assert!((safe_score(f64::INFINITY, 0.0)).abs() < 0.001); }
    #[test] fn test_safe_normal() { assert!((safe_score(0.75, 0.0) - 0.75).abs() < 0.001); }
    #[test] fn test_config_normalize() {
        let mut c = ScoreConfig { temporal_weight: 1.0, ghost_weight: 1.0, belief_weight: 1.0, domain_weight: 1.0, frequency_weight: 1.0 };
        c.normalize(); assert!((c.total_weight() - 1.0).abs() < 0.001);
    }
    #[test] fn test_zero_weights() {
        let c = ScoreConfig { temporal_weight: 0.0, ghost_weight: 0.0, belief_weight: 0.0, domain_weight: 0.0, frequency_weight: 0.0 };
        assert!((TileScorer::with_config(c).score(&fresh())).abs() < 0.001);
    }
    #[test] fn test_top_k_overflow() {
        let mut top = TopK::new(2);
        top.try_insert(ScoredTile { id: "a".into(), score: 0.3, signals: Default::default() });
        top.try_insert(ScoredTile { id: "b".into(), score: 0.5, signals: Default::default() });
        top.try_insert(ScoredTile { id: "c".into(), score: 0.9, signals: Default::default() });
        top.try_insert(ScoredTile { id: "d".into(), score: 0.1, signals: Default::default() });
        assert_eq!(top.len(), 2); assert_eq!(top.results()[0].id, "c");
    }
}
