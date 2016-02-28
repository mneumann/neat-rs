use rand::Rng;
use fitness::Fitness;

/// Measures the genetic distance. This can be applied on a variety of levels.
/// For example, this is used to measure the genetic distance (or compatibility)
/// of two genomes. But it is also used to measure the weighted distance between
/// two connection genes. So the actual meaning of the measure highly depends
/// on the concret types.
pub trait Distance<T> {
    fn distance(&self, left: &T, right: &T) -> f64;
}

pub trait Genotype: Send + Clone {}

/// Mates two individuals, producing one offspring.
/// There is no need to use both individuals. Instead it can also
/// be used mutation only. Usually this is either crossover or mutation,
/// or both.
pub trait Mate<T: Genotype> {
    fn mate<R: Rng>(&mut self,
                    parent_left: &T,
                    parent_right: &T,
                    prefer_mutate: bool,
                    rng: &mut R)
                    -> T;
}

/// Trait to calculate the fitness for a genome.

pub trait FitnessEval<T: Genotype>: Sync {
    fn fitness(&self, genome: &T) -> Fitness;
}
