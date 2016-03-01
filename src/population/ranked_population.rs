use traits::Genotype;
use population::population::{Population, PopulationWithRating, PopulationWithRank};
use population::rated_population::RatedPopulation;
use population::individual::Individual;

#[derive(Debug)]
pub struct RankedPopulation<T>
    where T: Genotype
{
    ranked_individuals: Vec<Individual<T>>,
}

impl<T> Population for RankedPopulation<T>
    where T: Genotype
{
    type Genome = T;

    fn individuals(&self) -> &[Individual<Self::Genome>] {
        &self.ranked_individuals
    }

    fn move_individuals(self) -> Vec<Individual<Self::Genome>> {
        self.ranked_individuals
    }
}

impl<T> PopulationWithRating for RankedPopulation<T> where T: Genotype {}
impl<T> PopulationWithRank for RankedPopulation<T> where T: Genotype {}

impl<T> RankedPopulation<T>
    where T: Genotype
{
    pub fn from_rated(rated_pop: RatedPopulation<T>) -> Self {
        // higher value of fitness means that the individual is fitter.
        let mut individuals = rated_pop.move_individuals();

        individuals.sort_by(|a, b| a.fitness().cmp(&b.fitness()).reverse());

        RankedPopulation { ranked_individuals: individuals }
    }
}
