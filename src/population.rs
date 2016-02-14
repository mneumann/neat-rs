use super::fitness::Fitness;
use super::traits::{Genotype, Distance};
use super::selection;
use rand::{Rng, Closed01};
use super::traits::Mate;
use std::marker::PhantomData;
use std::num::Zero;
use std::fmt::Debug;
use std::cmp;
use rayon::par_iter::*;
use super::prob::probabilistic_round;

#[derive(Debug)]
pub struct Individual<T: Debug> {
    fitness: Fitness,
    genome: Box<T>,
}

pub trait Rating {}
#[derive(Debug)]
pub struct Rated;
#[derive(Debug)]
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

    pub fn rate<F>(mut self, f: &F) -> Population<T, Rated>
        where F: Fn(&T) -> Fitness
    {
        for ind in self.individuals.iter_mut() {
            let fitness = f(&ind.genome);
            ind.fitness = fitness;
        }
        Population {
            individuals: self.individuals,
            _marker: PhantomData,
        }
    }

    pub fn rate_par<F>(mut self, f: &F) -> Population<T, Rated>
        where F: Sync + Fn(&T) -> Fitness
    {
        self.individuals.par_iter_mut().for_each(|ind| {
            let fitness = f(&ind.genome);
            ind.fitness = fitness;
        });

        Population {
            individuals: self.individuals,
            _marker: PhantomData,
        }
    }
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

    pub fn max_fitness(&self) -> Option<Fitness> {
        self.individuals.iter().max_by_key(|ind| ind.fitness).map(|i| i.fitness)
    }

    // higher value of fitness means that the individual is fitter.
    pub fn sort(mut self) -> Self {
        (&mut self.individuals).sort_by(|a, b| a.fitness.cmp(&b.fitness).reverse());
        self
    }

    /// Merge `self` with the first `n` individuals from population `other`.
    pub fn merge(&mut self, other: Population<T, Rated>, n: Option<usize>) {
        let len = other.individuals.len();
        self.individuals.extend(other.individuals.into_iter().take(n.unwrap_or(len)));
    }

    // We want to create a population of approximately `pop_size`. We do not care if there are
    // slightly more or slightly less individuals than that.
    //
    // 1. sort whole popluation into niches
    // 2. calculate the number of offspring for each niche.
    // 3. each niche produces offspring.
    //    - sort each niche according to the fitness value.
    //    - r% of the best genomes produce offspring
    //    - determine the elitist size. those are always copied into the new generation.
    //
    pub fn produce_offspring<C, M, R>(self,
                                      pop_size: usize,
                                      // how many of the best individuals of a niche are copied as-is into the
                                      // new population?
                                      elite_percentage: Closed01<f64>,
                                      // how many of the best individuals of a niche are selected for
                                      // reproduction?
                                      selection_percentage: Closed01<f64>,
                                      tournament_k: usize,
                                      compatibility_threshold: f64,
                                      compatibility: &C,
                                      mate: &mut M,
                                      rng: &mut R)
                                      -> (Population<T, Rated>, Population<T, Unrated>)
        where R: Rng,
              C: Distance<T>,
              M: Mate<T>
    {
        debug!("population size: {}", self.len());

        assert!(elite_percentage.0 <= selection_percentage.0); // XXX
        assert!(self.len() > 0);
        let niches = self.partition(rng, compatibility_threshold, compatibility);
        let num_niches = niches.len();

        debug!("number of niches: {}", num_niches);

        assert!(num_niches > 0);
        let total_mean: Fitness = niches.iter().map(|ind| ind.mean_fitness()).sum();
        assert!(total_mean.get() >= 0.0);

        let mut new_unrated_population: Population<T, Unrated> = Population::new();
        let mut new_rated_population: Population<T, Rated> = Population::new();

        for niche in niches.into_iter() {
            let percentage_of_population: f64 = if total_mean.get() == 0.0 {
                // all individuals have a fitness of 0.0.
                // we will equally allow each niche to procude offspring.
                1.0 / (num_niches as f64)
            } else {
                (niche.mean_fitness() / total_mean).get()
            };

            // calculate new size of niche, and size of elites, and selection size.
            assert!(percentage_of_population >= 0.0 && percentage_of_population <= 1.0);

            let niche_size = pop_size as f64 * percentage_of_population;

            // number of elitary individuals to copy from the old niche generation into the new.
            let elite_size =
                cmp::max(1,
                         probabilistic_round(niche_size * elite_percentage.0, rng) as usize);

            // number of offspring to produce.
            let offspring_size = probabilistic_round(niche_size * (1.0 - elite_percentage.0),
                                                     rng) as usize;

            // number of the best individuals to use for mating.
            let select_size = probabilistic_round(niche_size *
                                                  selection_percentage.0,
                                                  rng) as usize;

            let sorted_niche = niche.sort();

            let select_size = cmp::min(select_size, sorted_niche.len());

            // at first produce `offspring_size` individuals from the top `select_size`
            // individuals.
            if select_size > 0 {
                let mut n = offspring_size;
                while n > 0 {
                    let (parent1, parent2) =
                        selection::tournament_selection_fast2(rng,
                                                              &|i, j| {
                                                                  sorted_niche.compare_ij(i, j)
                                                              },
                                                              select_size,
                                                              cmp::min(select_size, tournament_k),
                                                              3);

                    let offspring = mate.mate(&sorted_niche.individuals[parent1].genome,
                                              &sorted_niche.individuals[parent2].genome,
                                              rng);
                    new_unrated_population.add_genome(Box::new(offspring));
                    n -= 1;
                }
            }

            // now copy the elites
            new_rated_population.merge(sorted_niche, Some(elite_size));
        }


        return (new_rated_population, new_unrated_population);
    }

    // Partitions the whole population into species (niches)
    fn partition<R, C>(self,
                       rng: &mut R,
                       compatibility_threshold: f64,
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
                    Some(probe) => {
                        compatibility.distance(&probe.genome, &ind.genome) < compatibility_threshold
                    }
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

pub struct Runner<'a, T, C, M, F>
    where T: Genotype + Debug,
          C: Distance<T> + 'a,
          M: Mate<T> + 'a,
          F: Sync + Fn(&T) -> Fitness + 'a
{
    // anticipated population size
    pub pop_size: usize,
    // how many of the best individuals of a niche are copied as-is into the
    // new population?
    pub elite_percentage: Closed01<f64>,
    // how many of the best individuals of a niche are selected for
    // reproduction?
    pub selection_percentage: Closed01<f64>,
    pub tournament_k: usize,
    pub compatibility_threshold: f64,
    pub compatibility: &'a C,
    pub mate: &'a mut M,
    pub fitness: &'a F,
    pub _marker: PhantomData<T>,
}

impl<'a, T, C, M, F> Runner<'a, T, C, M, F>
    where T: Genotype + Debug,
          C: Distance<T> + 'a,
          M: Mate<T> + 'a,
          F: Sync + Fn(&T) -> Fitness + 'a
{
    pub fn run<R, G>(&mut self,
                     initial_pop: Population<T, Unrated>,
                     goal_condition: &G,
                     rng: &mut R)
                     -> (usize, Population<T, Rated>)
        where R: Rng,
              G: Fn(usize, &Population<T, Rated>) -> bool
    {
        let mut iteration: usize = 0;
        let mut current_rated_pop = initial_pop.rate_par(self.fitness);

        while !goal_condition(iteration, &current_rated_pop) {
            let (new_rated, new_unrated) =
                current_rated_pop.produce_offspring(self.pop_size,
                                                    Closed01(self.elite_percentage.0),
                                                    Closed01(self.selection_percentage.0),
                                                    self.tournament_k,
                                                    self.compatibility_threshold,
                                                    self.compatibility,
                                                    self.mate,
                                                    rng);

            current_rated_pop = new_rated;
            current_rated_pop.merge(new_unrated.rate_par(self.fitness), None);
            iteration += 1;
        }

        return (iteration, current_rated_pop);
    }
}
