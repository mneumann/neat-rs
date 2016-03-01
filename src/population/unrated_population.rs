use population::population::Population;
use population::individual::Individual;
use traits::Genotype;

#[derive(Debug)]
pub struct UnratedPopulation<T>
    where T: Genotype
{
    unrated_individuals: Vec<Individual<T>>,
}

impl<T> Population for UnratedPopulation<T>
    where T: Genotype
{
    type Genome = T;

    fn individuals(&self) -> &[Individual<Self::Genome>] {
        &self.unrated_individuals
    }

    fn move_individuals(self) -> Vec<Individual<Self::Genome>> {
        self.unrated_individuals
    }
}

impl<T> UnratedPopulation<T>
    where T: Genotype
{
    pub fn add_unrated_individual(&mut self, ind: Individual<T>) {
        assert!(!ind.has_fitness());
        self.unrated_individuals.push(ind);
    }
}
