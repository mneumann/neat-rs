use traits::Genotype;
use population::individual::Individual;
use rand::Rng;
use std::cmp;

pub trait Population {
    type Genome: Genotype;

    fn individuals(&self) -> &[Individual<Self::Genome>];

    fn move_individuals(self) -> Vec<Individual<Self::Genome>>;

    /// Returns the number of individuals in the population

    fn len(&self) -> usize {
        self.individuals().len()
    }

    /// Returns a reference to a random individual of the population.

    fn random_individual<R: Rng>(&self, rng: &mut R) -> Option<&Individual<Self::Genome>> {
        rng.choose(self.individuals())
    }

    /// Returns a reference to two distinct random individuals of the population.

    fn two_random_distinct_individuals<R: Rng>
        (&self,
         rng: &mut R,
         retries: usize)
         -> Option<(&Individual<Self::Genome>, &Individual<Self::Genome>)> {
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

/// A ranked population is sorted (from better to worse) according to the individuals fitness values.

pub trait PopulationWithRank : PopulationWithRating {
    fn best_individual(&self) -> Option<&Individual<Self::Genome>> {
        self.individuals().first()
    }

    fn worst_individual(&self) -> Option<&Individual<Self::Genome>> {
        self.individuals().last()
    }

    /// Selects two random parents within the top `select_size` individuals. Tries to find distinct
    /// individuals. Retries `retries` time until giving up to find distinct individuals, in which
    /// case it returns a two equal indices. The first returned parent index has better performance
    /// (or equal).

    fn select_parent_indices<R>(&self,
                                select_size: usize,
                                retries: usize,
                                rng: &mut R)
                                -> (usize, usize)
        where R: Rng
    {
        let select_size = cmp::min(self.len(), select_size);
        assert!(select_size > 0);

        // We do not need tournament selection here as our population is sorted (ranked).
        // We simply determine two individuals out of `select_size`.

        let parent1 = rng.gen_range(0, select_size);
        let mut parent2 = rng.gen_range(0, select_size);

        // try to find a parent2 != parent1. retry `retries` times.
        for _ in 0..retries {
            if parent2 != parent1 {
                break;
            }
            parent2 = rng.gen_range(0, select_size);
        }

        // `mate` assumes that the first parent performs better.
        if parent1 < parent2 {
            (parent1, parent2)
        } else {
            (parent2, parent1)
        }
    }
}
