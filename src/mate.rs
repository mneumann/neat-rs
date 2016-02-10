use super::traits::Genotype;
use rand::Rng;

/// Mates two individuals, producing one offspring.
/// There is no need to use both individuals. Instead it can also
/// be used mutation only. Usually this is either crossover or mutation,
/// or both.
pub trait Mate<T: Genotype> {
    fn mate<R: Rng>(&self, parent_left: &T, parent_right: &T, rng: &mut R) -> T;
}
