use super::traits::Gene;

/// Align the innovations of two InnovationContainers.
#[derive(Debug, Clone)]
pub enum Alignment<'a, T: Gene + 'a> {
    Match(&'a T, &'a T),
    DisjointLeft(&'a T),
    DisjointRight(&'a T),
    ExcessLeft(&'a T),
    ExcessRight(&'a T),
}

#[cfg(test)]
impl<'a, T: Gene + 'a> Alignment<'a, T> {
    pub fn is_match(&self) -> bool {
        match self {
            &Alignment::Match(..) => true,
            _ => false,
        }
    }

    pub fn is_disjoint_left(&self) -> bool {
        match self {
            &Alignment::DisjointLeft(..) => true,
            _ => false,
        }
    }

    pub fn is_disjoint_right(&self) -> bool {
        match self {
            &Alignment::DisjointRight(..) => true,
            _ => false,
        }
    }

    pub fn is_disjoint(&self) -> bool {
        match self {
            &Alignment::DisjointLeft(..) |
            &Alignment::DisjointRight(..) => true,
            _ => false,
        }
    }

    pub fn is_excess_left(&self) -> bool {
        match self {
            &Alignment::ExcessLeft(..) => true,
            _ => false,
        }
    }

    pub fn is_excess_right(&self) -> bool {
        match self {
            &Alignment::ExcessRight(..) => true,
            _ => false,
        }
    }

    pub fn is_excess(&self) -> bool {
        match self {
            &Alignment::ExcessLeft(..) |
            &Alignment::ExcessRight(..) => true,
            _ => false,
        }
    }
}

pub struct AlignmentMetric {
    pub matching: usize,
    pub disjoint: usize,
    pub excess: usize,
    pub weight_distance: f64,
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
