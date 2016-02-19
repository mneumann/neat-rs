use innovation::Innovation;
use std::fmt::Debug;

// Equality and order of Gene's should be purely determined by their innovation numbers.
pub trait Gene: Clone + Debug {
    fn innovation(&self) -> Innovation;
    fn weight_distance(&self, other: &Self) -> f64;
}
