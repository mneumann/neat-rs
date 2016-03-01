use fitness::Fitness;
use traits::Genotype;

#[derive(Debug)]
pub struct Individual<T>
    where T: Genotype
{
    genome: Box<T>,
    fitness: Option<Fitness>,
}

impl<T> Individual<T>
    where T: Genotype
{
    pub fn new_unrated(genome: Box<T>) -> Self {
        Individual {
            genome: genome,
            fitness: None,
        }
    }

    pub fn has_fitness(&self) -> bool {
        self.fitness.is_some()
    }

    pub fn fitness(&self) -> Fitness {
        self.fitness.unwrap()
    }

    pub fn assign_fitness(&mut self, fitness: Fitness) {
        assert!(!self.has_fitness());

        self.fitness = Some(fitness);
    }

    pub fn genome(&self) -> &T {
        &self.genome
    }
}
