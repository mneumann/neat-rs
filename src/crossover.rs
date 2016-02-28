use prob::Prob;

/// A specific form of crossover where the probabilities below determine from
/// which parent a gene is taken.
///
/// XXX: It's probably faster to not use floats here.
#[derive(Debug, Copy, Clone)]
pub struct ProbabilisticCrossover {
    /// Probability to take a matching gene from the fitter (left) parent.
    pub prob_match_left: Prob,

    /// Probability to take a disjoint gene from the fitter (left) parent.
    pub prob_disjoint_left: Prob,

    /// Probability to take an excess gene from the fitter (left) parent.
    pub prob_excess_left: Prob,

    /// Probability to take a disjoint gene from the less fit (right) parent.
    pub prob_disjoint_right: Prob,

    /// Probability to take an excess gene from the less fit (right) parent.
    pub prob_excess_right: Prob,
}
