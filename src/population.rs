use super::fitness::Fitness;
use super::traits::{Genotype, Distance};
use super::selection;
use rand::{Rng, Closed01};
use super::mate::Mate;
use std::marker::PhantomData;
use std::num::Zero;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Individual<T: Debug> {
    fitness: Fitness,
    genome: Box<T>,
}

pub trait Rating {}
pub struct Rated;
pub struct Unrated;

impl Rating for Rated {}
impl Rating for Unrated {}

#[derive(Debug)]
pub struct Population<T: Genotype + Debug, R: Rating> {
    individuals: Vec<Individual<T>>,
    _marker: PhantomData<R>,
}

impl<T: Genotype + Debug, R: Rating> Population<T, R> {
    pub fn new() -> Population<T, R> {
        Population {
            individuals: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn new_from_vec(v: Vec<Individual<T>>) -> Population<T, R> {
        Population {
            individuals: v,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }
}

impl<T: Genotype + Debug> Population<T, Unrated> {
    pub fn add_genome(&mut self, genome: Box<T>) {
        self.individuals.push(Individual {
            fitness: Zero::zero(),
            genome: genome,
        });
    }

    // pub fn rate<F>(self, f: &F) -> RatedPopulation<T> where F: Fn(&T) -> Fitness {
    // genomes: Vec<(Fitness, Box<T>)>,
    // }
    //
}

impl<T: Genotype + Debug> Population<T, Rated> {
    pub fn add_individual(&mut self, ind: Individual<T>) {
        self.individuals.push(ind);
    }

    // Return true if genome at position `i` is better that `j`
    fn compare_ij(&self, i: usize, j: usize) -> bool {
        self.individuals[i].fitness > self.individuals[j].fitness
    }

    fn mean_fitness(&self) -> Fitness {
        let sum: Fitness = self.individuals.iter().map(|ind| ind.fitness).sum();
        sum / Fitness::new(self.len() as f64)
    }

    fn sort(mut self) -> Self {
        (&mut self.individuals).sort_by(|a, b| a.fitness.cmp(&b.fitness));
        self
    }

    /// Merge `self` with the first `n` individuals from population `other`.
    fn merge(&mut self, other: Population<T, Rated>, n: usize) {
        self.individuals.extend(other.individuals.into_iter().take(n));
    }

    // We want to create a population of approximately ```pop_size```.
    // We do not care if there are slightly more individuals than that.
    //
    // 1. sort whole popluation into niches
    // 2. calculate the number of offspring for each niche.
    // 3. each niche produces offspring.
    //    - sort each niche according to the fitness value.
    //    - determine the elitist size. those are always copied into the new generation.
    //    - r% of the best genomes produce offspring
    //
    fn produce_offspring<R, C, M>(self,
                                  pop_size: usize,
                                  // how many of the best individuals of a niche are copied as-is into the
                                  // new population?
                                  elite_percentage: Closed01<f64>,
                                  // how many of the best individuals of a niche are selected for
                                  // reproduction?
                                  selection_percentage: Closed01<f64>,
                                  tournament_k: usize,
                                  rng: &mut R,
                                  threshold: f64,
                                  compatibility: &C,
                                  mate: &M)
                                  -> (Population<T, Rated>, Population<T, Unrated>)
        where R: Rng,
              C: Distance<T>,
              M: Mate<T>
    {
        assert!(elite_percentage.0 <= selection_percentage.0); // XXX
        assert!(self.len() > 0);
        let niches = self.partition(rng, threshold, compatibility);
        assert!(niches.len() > 0);
        let total_mean: Fitness = niches.iter().map(|ind| ind.mean_fitness()).sum();
        // XXX: total_mean = 0.0?

        let mut new_unrated_population: Population<T, Unrated> = Population::new();
        let mut new_rated_population: Population<T, Rated> = Population::new();

        for niche in niches.into_iter() {
            // calculate new size of niche, and size of elites, and selection size.
            let percentage_of_population: f64 = (niche.mean_fitness() / total_mean).get();
            assert!(percentage_of_population >= 0.0 && percentage_of_population <= 1.0);

            let niche_size = pop_size as f64 * percentage_of_population;

            // number of elitary individuals to copy from the old niche generation into the new.
            let elite_size = (niche_size * elite_percentage.0).round() as usize;

            // number of offspring to produce.
            let offspring_size = (niche_size * (1.0 - elite_percentage.0)).round() as usize;

            // number of the best individuals to use for mating.
            let select_size = (niche_size * selection_percentage.0).round() as usize;

            let sorted_niche = niche.sort();

            // at first produce ```offspring_size``` individuals from the top ```select_size```
            // individuals.
            if select_size > 0 {
                let mut n = offspring_size;
                while n > 0 {
                    let parent1 = selection::tournament_selection_fast(rng,
                                                                       |i, j| {
                                                                           sorted_niche.compare_ij(i,j)
                                                                       },
                                                                       select_size,
                                                                       tournament_k);
                    let parent2 = selection::tournament_selection_fast(rng,
                                                                       |i, j| {
                                                                           sorted_niche.compare_ij(i,j)
                                                                       },
                                                                       select_size,
                                                                       tournament_k);

                    let offspring = mate.mate(&sorted_niche.individuals[parent1].genome,
                                              &sorted_niche.individuals[parent2].genome,
                                              rng);
                    // let offspring = sorted_niche.genomes[parent1].1.clone();
                    new_unrated_population.add_genome(Box::new(offspring));
                    n -= 1;
                }
            }

            // now copy the elites
            new_rated_population.merge(sorted_niche, elite_size);
        }


        return (new_rated_population, new_unrated_population);
    }

    // Partitions the whole population into species (niches)
    fn partition<R, C>(self,
                       rng: &mut R,
                       threshold: f64,
                       compatibility: &C)
                       -> Vec<Population<T, Rated>>
        where R: Rng,
              C: Distance<T>
    {
        let mut niches: Vec<Population<T, Rated>> = Vec::new();

        'outer: for ind in self.individuals.into_iter() {
            for niche in niches.iter_mut() {
                // Is this genome compatible with this niche? Test against a random genome.
                let compatible = match rng.choose(&niche.individuals) {
                    Some(probe) => compatibility.distance(&probe.genome, &ind.genome) < threshold,
                    // If a niche is empty, a genome always is compatible (note that a niche can't be empyt)
                    None => true,
                };
                if compatible {
                    niche.add_individual(ind);
                    continue 'outer;
                }
            }
            // if no compatible niche was found, create a new niche containing this genome.
            let mut new_niche = Population::new();
            new_niche.add_individual(ind);
            niches.push(new_niche);
        }

        niches
    }
}
