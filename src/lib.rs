//! plato-tile-scorer v2 — Multi-signal scoring with controversy signal
//! From DeepSeek's insight: "Evolution needs predators"
//! Tiles that survive counterpoints are MORE reliable than unchallenged tiles

/// Scoring signals with configurable weights
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub temporal: f64,
    pub ghost: f64,
    pub belief: f64,
    pub domain: f64,
    pub frequency: f64,
    pub keyword: f64,
    pub controversy: f64,    // NEW: counterpoint survival
    pub usage_quality: f64,  // NEW: success rate weighting
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            temporal: 0.10,
            ghost: 0.10,
            belief: 0.20,
            domain: 0.15,
            frequency: 0.10,
            keyword: 0.25,
            controversy: 0.05,
            usage_quality: 0.05,
        }
    }
}

/// Input signals for scoring a single tile
#[derive(Debug, Clone)]
pub struct ScoringInput {
    pub query: String,
    pub tile_content: String,
    pub tile_question: String,

    // v1 signals
    pub temporal_score: f64,       // 0.0-1.0 (freshness)
    pub ghost_score: f64,          // 0.0-1.0 (resurrection priority)
    pub belief_score: f64,         // 0.0-1.0 (DCS consensus)
    pub domain_relevance: f64,     // 0.0-1.0 (domain match)
    pub access_frequency: f64,     // 0.0-1.0 (normalized)
    pub keyword_match: f64,        // 0.0-1.0

    // v2 signals (from JC1's research)
    pub controversy_score: f64,    // 0.0-1.0 (counterpoints survived)
    pub usage_count: u64,          // raw count
    pub success_rate: f64,         // 0.0-1.0
    pub confidence: f64,           // 0.0-1.0
    pub has_counterpoints: bool,
    pub is_challenged: bool,       // has at least 1 counterpoint
    pub tile_age_seconds: u64,
}

impl ScoringInput {
    pub fn minimal(query: &str, content: &str) -> Self {
        Self {
            query: query.to_string(),
            tile_content: content.to_string(),
            tile_question: String::new(),
            temporal_score: 1.0,
            ghost_score: 0.0,
            belief_score: 0.5,
            domain_relevance: 0.5,
            access_frequency: 0.0,
            keyword_match: 0.5,
            controversy_score: 0.0,
            usage_count: 0,
            success_rate: 1.0,
            confidence: 0.5,
            has_counterpoints: false,
            is_challenged: false,
            tile_age_seconds: 0,
        }
    }
}

/// Score result
#[derive(Debug, Clone)]
pub struct ScoreResult {
    pub total: f64,
    pub signals: SignalBreakdown,
    pub gated: bool,
    pub gate_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SignalBreakdown {
    pub temporal: f64,
    pub ghost: f64,
    pub belief: f64,
    pub domain: f64,
    pub frequency: f64,
    pub keyword: f64,
    pub controversy: f64,
    pub usage_quality: f64,
}

/// Multi-signal tile scorer
pub struct TileScorer {
    weights: ScoringWeights,
    keyword_gate: f64,       // Below this → score 0.0
    controversy_floor: f64,  // Min controversy for unchallenged tiles
}

impl TileScorer {
    pub fn new() -> Self {
        Self {
            weights: ScoringWeights::default(),
            keyword_gate: 0.01,
            controversy_floor: 0.3,
        }
    }

    pub fn with_weights(weights: ScoringWeights) -> Self {
        Self {
            weights,
            keyword_gate: 0.01,
            controversy_floor: 0.3,
        }
    }

    /// Score a single tile
    pub fn score(&self, input: &ScoringInput) -> ScoreResult {
        // Keyword gate: if keyword match is too low, tile is irrelevant
        if input.keyword_match < self.keyword_gate {
            return ScoreResult {
                total: 0.0,
                signals: SignalBreakdown::zero(),
                gated: true,
                gate_reason: Some("keyword_match_below_gate".to_string()),
            };
        }

        // Controversy signal: unchallenged tiles get a floor value
        // DeepSeek's insight: untested reliability < tested reliability
        let controversy = if input.has_counterpoints {
            input.controversy_score
        } else {
            // Unchallenged tiles: penalize slightly (unknown reliability)
            self.controversy_floor * input.confidence
        };

        // Usage quality: success rate weighted by usage volume
        // A tile used 100 times at 90% success is better than used once at 100%
        let usage_factor = (input.usage_count as f64 + 1.0).ln() / 10.0; // Normalize ln
        let usage_quality = input.success_rate * usage_factor.min(1.0);

        // Compute weighted signals
        let temporal = self.weights.temporal * input.temporal_score;
        let ghost = self.weights.ghost * input.ghost_score;
        let belief = self.weights.belief * input.belief_score;
        let domain = self.weights.domain * input.domain_relevance;
        let frequency = self.weights.frequency * input.access_frequency;
        let keyword = self.weights.keyword * input.keyword_match;
        let controversy_w = self.weights.controversy * controversy;
        let usage_w = self.weights.usage_quality * usage_quality;

        let total = temporal + ghost + belief + domain + frequency + keyword + controversy_w + usage_w;

        ScoreResult {
            total,
            signals: SignalBreakdown {
                temporal: input.temporal_score,
                ghost: input.ghost_score,
                belief: input.belief_score,
                domain: input.domain_relevance,
                frequency: input.access_frequency,
                keyword: input.keyword_match,
                controversy,
                usage_quality,
            },
            gated: false,
            gate_reason: None,
        }
    }

    /// Score and rank multiple tiles
    pub fn rank(&self, inputs: &[ScoringInput]) -> Vec<(usize, ScoreResult)> {
        let mut results: Vec<(usize, ScoreResult)> = inputs
            .iter()
            .enumerate()
            .map(|(i, input)| (i, self.score(input)))
            .collect();
        results.sort_by(|a, b| b.1.total.partial_cmp(&a.1.total).unwrap());
        results
    }

    /// Score with deadband priority boost
    pub fn score_with_deadband(&self, input: &ScoringInput, priority: &str) -> ScoreResult {
        let mut result = self.score(input);
        match priority {
            "P0" => result.total += 10.0,
            "P1" => result.total += 1.0,
            _ => {}
        }
        result
    }

    /// Top-N selection
    pub fn top_n(&self, inputs: &[ScoringInput], n: usize) -> Vec<(usize, ScoreResult)> {
        self.rank(inputs).into_iter().take(n).collect()
    }
}

impl SignalBreakdown {
    pub fn zero() -> Self {
        Self {
            temporal: 0.0, ghost: 0.0, belief: 0.0, domain: 0.0,
            frequency: 0.0, keyword: 0.0, controversy: 0.0, usage_quality: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(query: &str, keyword: f64, confidence: f64) -> ScoringInput {
        let mut input = ScoringInput::minimal(query, "tile content");
        input.keyword_match = keyword;
        input.confidence = confidence;
        input
    }

    #[test]
    fn test_basic_scoring() {
        let scorer = TileScorer::new();
        let input = make_input("rust programming", 0.8, 0.9);
        let result = scorer.score(&input);
        assert!(result.total > 0.0);
        assert!(!result.gated);
    }

    #[test]
    fn test_keyword_gate_blocks() {
        let scorer = TileScorer::new();
        let mut input = make_input("quantum physics", 0.001, 0.9);
        input.keyword_match = 0.001;
        let result = scorer.score(&input);
        assert_eq!(result.total, 0.0);
        assert!(result.gated);
        assert_eq!(result.gate_reason, Some("keyword_match_below_gate".to_string()));
    }

    #[test]
    fn test_keyword_gate_passes() {
        let scorer = TileScorer::new();
        let mut input = make_input("rust", 0.02, 0.9);
        let result = scorer.score(&input);
        assert!(result.total > 0.0);
        assert!(!result.gated);
    }

    #[test]
    fn test_controversy_boost() {
        let scorer = TileScorer::new();
        // Challenged tile with high controversy and high usage
        let mut challenged = make_input("rust", 0.8, 0.9);
        challenged.has_counterpoints = true;
        challenged.controversy_score = 0.9;
        challenged.usage_count = 50;
        challenged.success_rate = 0.95;
        // Unchallenged tile with same base
        let mut unchallenged = make_input("rust", 0.8, 0.9);
        unchallenged.has_counterpoints = false;
        unchallenged.usage_count = 50;
        unchallenged.success_rate = 0.95;

        let r1 = scorer.score(&challenged);
        let r2 = scorer.score(&unchallenged);
        // Challenged tile should score higher (tested reliability)
        assert!(r1.total > r2.total, "challenged ({}) should beat unchallenged ({})", r1.total, r2.total);
    }

    #[test]
    fn test_usage_quality_signal() {
        let scorer = TileScorer::new();
        // Highly used, high success rate
        let mut popular = make_input("rust", 0.8, 0.9);
        popular.usage_count = 100;
        popular.success_rate = 0.95;
        // Rarely used
        let mut rare = make_input("rust", 0.8, 0.9);
        rare.usage_count = 1;
        rare.success_rate = 1.0;

        let r1 = scorer.score(&popular);
        let r2 = scorer.score(&rare);
        // Popular with high success should score higher
        assert!(r1.signals.usage_quality > r2.signals.usage_quality);
    }

    #[test]
    fn test_ranking_order() {
        let scorer = TileScorer::new();
        let inputs = vec![
            make_input("rust", 0.3, 0.5),
            make_input("rust", 0.9, 0.9),
            make_input("rust", 0.6, 0.7),
        ];
        let ranked = scorer.rank(&inputs);
        assert_eq!(ranked[0].0, 1); // Highest keyword+confidence first
    }

    #[test]
    fn test_top_n() {
        let scorer = TileScorer::new();
        let inputs = vec![
            make_input("a", 0.3, 0.5),
            make_input("b", 0.9, 0.9),
            make_input("c", 0.6, 0.7),
            make_input("d", 0.8, 0.8),
        ];
        let top2 = scorer.top_n(&inputs, 2);
        assert_eq!(top2.len(), 2);
        assert!(top2[0].1.total >= top2[1].1.total);
    }

    #[test]
    fn test_deadband_priority_boost() {
        let scorer = TileScorer::new();
        let input = make_input("rust", 0.5, 0.8);
        let normal = scorer.score(&input);
        let p0 = scorer.score_with_deadband(&input, "P0");
        let p1 = scorer.score_with_deadband(&input, "P1");
        assert!((p0.total - normal.total - 10.0).abs() < 0.01);
        assert!((p1.total - normal.total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_signal_breakdown() {
        let scorer = TileScorer::new();
        let input = make_input("rust", 0.5, 0.8);
        let result = scorer.score(&input);
        assert!(result.signals.keyword > 0.0);
        assert!(result.signals.temporal >= 0.0);
        assert!(result.signals.controversy >= 0.0);
        assert!(result.signals.usage_quality >= 0.0);
    }

    #[test]
    fn test_custom_weights() {
        let weights = ScoringWeights {
            temporal: 0.0, ghost: 0.0, belief: 0.0, domain: 0.0,
            frequency: 0.0, keyword: 1.0, controversy: 0.0, usage_quality: 0.0,
        };
        let scorer = TileScorer::with_weights(weights);
        let input = make_input("rust", 0.7, 0.5);
        let result = scorer.score(&input);
        assert!((result.total - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_controversy_floor_for_unchallenged() {
        let scorer = TileScorer::new();
        let mut input = make_input("rust", 0.8, 1.0);
        input.has_counterpoints = false;
        let result = scorer.score(&input);
        // Floor = 0.3 * 1.0 = 0.3, weighted by 0.05 = 0.015
        assert!(result.signals.controversy > 0.0);
    }

    #[test]
    fn test_controversy_signal_contributes() {
        let scorer = TileScorer::new();
        // Same base tile, but challenged with controversy
        let mut challenged = make_input("rust", 0.8, 0.9);
        challenged.has_counterpoints = true;
        challenged.controversy_score = 1.0;
        let mut unchallenged = make_input("rust", 0.8, 0.9);
        unchallenged.has_counterpoints = false;
        let r1 = scorer.score(&challenged);
        let r2 = scorer.score(&unchallenged);
        // Challenged tile should have higher controversy signal
        assert!(r1.signals.controversy > r2.signals.controversy);
    }

    #[test]
    fn test_zero_confidence_scores_low() {
        let scorer = TileScorer::new();
        let mut input = make_input("rust", 0.5, 0.0);
        input.has_counterpoints = false;
        let result = scorer.score(&input);
        // With 0 confidence: controversy floor = 0, usage_quality = 0
        // But temporal(1.0*0.1) + belief(0.5*0.2) + domain(0.5*0.15) + keyword(0.5*0.25) = 0.1+0.1+0.075+0.125 = 0.4
        // That's still modest compared to a high-confidence tile
        assert!(result.total < 0.5, "total should be modest, got {}", result.total);
    }

    #[test]
    fn test_all_signals_contribute() {
        let scorer = TileScorer::new();
        let mut input = ScoringInput::minimal("test", "content");
        input.temporal_score = 1.0;
        input.ghost_score = 1.0;
        input.belief_score = 1.0;
        input.domain_relevance = 1.0;
        input.access_frequency = 1.0;
        input.keyword_match = 1.0;
        input.has_counterpoints = true;
        input.controversy_score = 1.0;
        input.usage_count = 1000;
        input.success_rate = 1.0;
        input.confidence = 1.0;
        let result = scorer.score(&input);
        assert!(result.total > 0.5);
    }

    #[test]
    fn test_gated_result_has_zero_breakdown() {
        let scorer = TileScorer::new();
        let mut input = make_input("q", 0.001, 0.9);
        let result = scorer.score(&input);
        assert_eq!(result.signals.keyword, 0.0);
        assert_eq!(result.signals.temporal, 0.0);
    }

    #[test]
    fn test_empty_query() {
        let scorer = TileScorer::new();
        let input = make_input("", 0.5, 0.8);
        let result = scorer.score(&input);
        assert!(result.total >= 0.0);
    }
}
