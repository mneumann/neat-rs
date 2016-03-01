use population::population::{Population, PopulationWithRating};
use population::unrated_population::UnratedPopulation;
use population::individual::Individual;
use traits::{Genotype, FitnessEval};
use rayon::par_iter::*;

#[derive(Debug)]
pub struct RatedPopulation<T>
    where T: Genotype
{
    rated_individuals: Vec<Individual<T>>,
}

impl<T> Population for RatedPopulation<T>
    where T: Genotype
{
    type Genome = T;

    fn individuals(&self) -> &[Individual<Self::Genome>] {
        &self.rated_individuals
    }

    fn move_individuals(self) -> Vec<Individual<Self::Genome>> {
        self.rated_individuals
    }
}

impl<T> PopulationWithRating for RatedPopulation<T> where T: Genotype {}

impl<T> RatedPopulation<T>
    where T: Genotype
{
    pub fn from_unrated_seq<F>(unrated_pop: UnratedPopulation<T>, f: &F) -> Self
        where F: FitnessEval<T>
    {
        let mut individuals = unrated_pop.move_individuals();

        for ind in individuals.iter_mut() {
            let fitness = f.fitness(&ind.genome());
            ind.assign_fitness(fitness);
        }

        RatedPopulation { rated_individuals: individuals }
    }

    pub fn from_unrated_par<F>(unrated_pop: UnratedPopulation<T>, f: &F) -> Self
        where F: FitnessEval<T>
    {
        let mut individuals = unrated_pop.move_individuals();

        individuals.par_iter_mut().for_each(|ind| {
            let fitness = f.fitness(&ind.genome());
            ind.assign_fitness(fitness);
        });

        RatedPopulation { rated_individuals: individuals }
    }

    pub fn add_rated_individual(&mut self, ind: Individual<T>) {
        assert!(ind.has_fitness());
        self.rated_individuals.push(ind);
    }

    /// Append all individuals of population `other`.

    pub fn append(&mut self, other: RatedPopulation<T>) {
        self.rated_individuals.extend(other.rated_individuals.into_iter());
    }
}
