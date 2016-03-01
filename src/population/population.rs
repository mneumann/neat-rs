use traits::Genotype;
use population::individual::Individual;
use rand::Rng;

pub trait Population {
    type Genome: Genotype;

    fn individuals(&self) -> &[Individual<Self::Genome>];

    fn move_individuals(self) -> Vec<Individual<Self::Genome>>;

    /// Returns the number of individuals in the population

    fn len(&self) -> usize {
        self.individuals().len()
    }

    /// Returns a reference to a random individual of the population.

    fn random_individual<R: Rng>(&self, rng: &mut R) -> &Individual<Self::Genome> {
        rng.choose(self.individuals()).unwrap()
    }

    /// Returns a reference to two distinct random individuals of the population.

    fn two_random_distinct_individuals<R: Rng>(&self, rng: &mut R, retries: usize) -> Option<(&Individual<Self::Genome>, &Individual<Self::Genome>)> {
        if self.len() >= 2 {
            let a = rng.gen_range(0, self.len());
            for _ in 0..retries {
                let b = rng.gen_range(0, self.len());
                if b != a {
                    return Some((&self.individuals()[a], &self.individuals()[b]));
                }
            }
        }
        None
    }
}


pub trait PopulationWithRating: Population {
    fn best_individual(&self) -> Option<&Individual<Self::Genome>> {
        self.individuals().iter().max_by_key(|ind| ind.fitness())
    }

    fn worst_individual(&self) -> Option<&Individual<Self::Genome>> {
        self.individuals().iter().min_by_key(|ind| ind.fitness())
    }
}
