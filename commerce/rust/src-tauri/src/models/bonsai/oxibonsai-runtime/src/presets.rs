//! Predefined sampling parameter presets for common use cases.

use crate::sampling::SamplingParams;

/// Named preset for sampling behavior.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::presets::SamplingPreset;
/// use oxibonsai_runtime::sampling::SamplingParams;
///
/// // Use a preset directly
/// let params: SamplingParams = SamplingPreset::Balanced.into();
/// assert!((params.temperature - 0.7).abs() < f32::EPSILON);
///
/// // Iterate over all presets
/// for preset in SamplingPreset::all() {
///     let p = preset.params();
///     assert!(p.temperature >= 0.0);
///     assert!(p.top_p >= 0.0 && p.top_p <= 1.0);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingPreset {
    /// General use: temp=0.7, top_p=0.9, top_k=40, rep_pen=1.1
    Balanced,
    /// Creative writing: temp=1.0, top_p=0.95, top_k=0 (disabled), rep_pen=1.0
    Creative,
    /// Factual/code generation: temp=0.1, top_p=0.5, top_k=10, rep_pen=1.2
    Precise,
    /// Deterministic output: temp=0.0, greedy decoding
    Greedy,
    /// Chat/conversation: temp=0.8, top_p=0.9, top_k=50, rep_pen=1.1
    Conversational,
}

/// All available presets in a static array.
static ALL_PRESETS: [SamplingPreset; 5] = [
    SamplingPreset::Balanced,
    SamplingPreset::Creative,
    SamplingPreset::Precise,
    SamplingPreset::Greedy,
    SamplingPreset::Conversational,
];

impl SamplingPreset {
    /// Get the sampling parameters for this preset.
    pub fn params(&self) -> SamplingParams {
        match self {
            Self::Balanced => SamplingParams {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                repetition_penalty: 1.1,
                ..SamplingParams::default()
            },
            Self::Creative => SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.95,
                repetition_penalty: 1.0,
                ..SamplingParams::default()
            },
            Self::Precise => SamplingParams {
                temperature: 0.1,
                top_k: 10,
                top_p: 0.5,
                repetition_penalty: 1.2,
                ..SamplingParams::default()
            },
            Self::Greedy => SamplingParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                ..SamplingParams::default()
            },
            Self::Conversational => SamplingParams {
                temperature: 0.8,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                ..SamplingParams::default()
            },
        }
    }

    /// Human-readable name of this preset.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Balanced => "Balanced",
            Self::Creative => "Creative",
            Self::Precise => "Precise",
            Self::Greedy => "Greedy",
            Self::Conversational => "Conversational",
        }
    }

    /// Description of this preset's intended use case.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Balanced => "General-purpose: moderate creativity with good coherence",
            Self::Creative => "Creative writing: high diversity and novel outputs",
            Self::Precise => "Factual/code: low randomness for accurate outputs",
            Self::Greedy => "Deterministic: always picks the most likely token",
            Self::Conversational => "Chat: natural-sounding conversation with personality",
        }
    }

    /// Get all available presets.
    pub fn all() -> &'static [SamplingPreset] {
        &ALL_PRESETS
    }
}

impl From<SamplingPreset> for SamplingParams {
    fn from(preset: SamplingPreset) -> Self {
        preset.params()
    }
}

impl std::fmt::Display for SamplingPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_preset() {
        let params = SamplingPreset::Balanced.params();
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 40);
        assert!((params.top_p - 0.9).abs() < f32::EPSILON);
        assert!((params.repetition_penalty - 1.1).abs() < f32::EPSILON);
    }

    #[test]
    fn creative_preset() {
        let params = SamplingPreset::Creative.params();
        assert!((params.temperature - 1.0).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 0); // disabled
        assert!((params.top_p - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn precise_preset() {
        let params = SamplingPreset::Precise.params();
        assert!((params.temperature - 0.1).abs() < f32::EPSILON);
        assert!((params.repetition_penalty - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn greedy_preset() {
        let params = SamplingPreset::Greedy.params();
        assert!(params.temperature < f32::EPSILON);
    }

    #[test]
    fn conversational_preset() {
        let params = SamplingPreset::Conversational.params();
        assert!((params.temperature - 0.8).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 50);
    }

    #[test]
    fn all_presets_covers_all_variants() {
        let all = SamplingPreset::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&SamplingPreset::Balanced));
        assert!(all.contains(&SamplingPreset::Creative));
        assert!(all.contains(&SamplingPreset::Precise));
        assert!(all.contains(&SamplingPreset::Greedy));
        assert!(all.contains(&SamplingPreset::Conversational));
    }

    #[test]
    fn preset_names_non_empty() {
        for preset in SamplingPreset::all() {
            assert!(!preset.name().is_empty());
            assert!(!preset.description().is_empty());
        }
    }

    #[test]
    fn preset_into_sampling_params() {
        let params: SamplingParams = SamplingPreset::Balanced.into();
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn preset_display() {
        assert_eq!(format!("{}", SamplingPreset::Balanced), "Balanced");
        assert_eq!(format!("{}", SamplingPreset::Greedy), "Greedy");
    }

    #[test]
    fn all_presets_produce_valid_params() {
        for preset in SamplingPreset::all() {
            let params = preset.params();
            assert!(params.temperature >= 0.0);
            assert!(params.top_p >= 0.0 && params.top_p <= 1.0);
            assert!(params.repetition_penalty >= 1.0);
        }
    }

    #[test]
    fn preset_clone_and_copy() {
        let p = SamplingPreset::Creative;
        let p2 = p;
        let p3 = p;
        assert_eq!(p, p2);
        assert_eq!(p, p3);
    }
}
