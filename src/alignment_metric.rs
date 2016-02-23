#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AlignmentMetric {
    pub matching: usize,
    pub disjoint: usize,
    pub excess: usize,
    pub weight_distance: f64,
}

impl Eq for AlignmentMetric {
}

impl AlignmentMetric {
    pub fn new() -> AlignmentMetric {
        AlignmentMetric {
            matching: 0,
            disjoint: 0,
            excess: 0,
            weight_distance: 0.0,
        }
    }
}
