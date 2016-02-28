use super::fitness::Fitness;
use super::traits::{Genotype, Distance};
use rand::Rng;
use closed01::Closed01;
use super::traits::Mate;
use std::marker::PhantomData;
use std::fmt::Debug;
use std::cmp;
use std::mem;
use rayon::par_iter::*;
use super::prob::probabilistic_round;

#[derive(Debug)]
pub struct Individual<T: Debug + Genotype> {
    fitness: Option<Fitness>,
    genome: Box<T>,
}

impl<T: Debug + Genotype> Individual<T> {
    pub fn fitness(&self) -> Fitness {
        self.fitness.unwrap()
    }

    pub fn genome(&self) -> &T {
        &self.genome
    }
}

pub trait Rating { }

pub trait IsRated : Rating { }
pub trait IsRatedSorted : IsRated { }

#[derive(Debug)]
pub struct Unrated;

#[derive(Debug)]
pub struct Rated;

#[derive(Debug)]
pub struct RatedSorted;

impl Rating for Unrated {}
impl Rating for Rated {}
impl Rating for RatedSorted {}

impl IsRated for Rated {}
impl IsRated for RatedSorted {}
impl IsRatedSorted for RatedSorted {}

#[derive(Debug)]
pub struct Population<T: Genotype + Debug, R: Rating> {
    individuals: Vec<Individual<T>>,
    _marker: PhantomData<R>,
}

#[derive(Debug)]
pub struct Niches<T: Genotype + Debug> {
    total_individuals: usize,
    niches: Vec<Population<T, Rated>>,
}

impl<T: Genotype + Debug> Niches<T> {
    pub fn new() -> Self {
        Niches {
            total_individuals: 0,
            niches: Vec::new(),
        }
    }

    /// Creates a single niche from a `Population`.

    pub fn single(pop: Population<T, Rated>) -> Self {
        Niches {
            total_individuals: pop.len(),
            niches: vec![pop]
        }
    }

    /// Collapse all niches into a single `Population`.

    pub fn collapse(self) -> Population<T, Rated> {
        assert!(!self.niches.is_empty());
        let tot = self.total_individuals;

        let mut iter = self.niches.into_iter();
        let mut pop = iter.next().unwrap();

        for niche in iter {
            pop.append(niche);
        }

        assert!(tot == pop.len());
        pop
    }

    pub fn into_iter(self) -> ::std::vec::IntoIter<Population<T, Rated>> {
        self.niches.into_iter()
    }

    /// The sum of all "mean fitnesses" of all niches.

    fn total_mean(&self) -> Fitness {
        self.niches.iter().map(|ind| ind.mean_fitness()).sum()
    }

    /// Total number of individuals

    pub fn num_individuals(&self) -> usize {
        self.total_individuals
    }

    /// Number of niches

    pub fn num_niches(&self) -> usize {
        self.niches.len()
    }

    /// Add an individual to one of the niches.

    pub fn add_individual<R, C>(&mut self,
                                ind: Individual<T>,
                                compatibility_threshold: f64,
                                compatibility: &C,
                                rng: &mut R)
        where R: Rng,
              C: Distance<T>
    {
        self.total_individuals += 1;

        for niche in self.niches.iter_mut() {
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
                return;
            }
        }
        // if no compatible niche was found, create a new niche containing this genome.
        let mut new_niche = Population::new();
        new_niche.add_individual(ind);
        self.niches.push(new_niche);
    }

    /// Reproduce individuals of all niches. Each niche is allowed to reproduce a number of
    /// individuals relative to it's performance to other niches.
    ///
    /// All new individuals are put into a global population (actually it's two, one rated and
    /// one unrated).

    pub fn reproduce_global<M, R>(self,
                              new_pop_size: usize,
                              // how many of the best individuals of a niche are copied as-is into the
                              // new population?
                              elite_percentage: Closed01<f64>,
                              // how many of the best individuals of a niche are selected for
                              // reproduction?
                              selection_percentage: Closed01<f64>,
                              mate: &mut M,
                              rng: &mut R)
                              -> (Population<T, Rated>, Population<T, Unrated>)
        where M: Mate<T>,
              R: Rng
    {
        assert!(self.num_individuals() > 0);
        assert!(self.num_niches() > 0);
        assert!(elite_percentage <= selection_percentage); // XXX

        let num_niches = self.num_niches();
        let total_mean = self.total_mean();

        assert!(total_mean.get() >= 0.0);

        let mut new_unrated_population: Population<T, Unrated> = Population::new();
        let mut new_rated_population: Population<T, Rated> = Population::new();

        for niche in self.into_iter() {
            let percentage_of_population: f64 = if total_mean.get() == 0.0 {
                // all individuals have a fitness of 0.0.
                // we will equally allow each niche to procude offspring.
                1.0 / (num_niches as f64)
            } else {
                (niche.mean_fitness() / total_mean).get()
            };

            // calculate new size of niche, and size of elites, and selection size.
            assert!(percentage_of_population >= 0.0 && percentage_of_population <= 1.0);

            let niche_size = new_pop_size as f64 * percentage_of_population;

            niche.reproduce_into(niche_size, elite_percentage, selection_percentage, mate,
                            &mut new_unrated_population, &mut new_rated_population, rng);
        }

        return (new_rated_population, new_unrated_population);
    }
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
            fitness: None,
            genome: genome,
        });
    }

    pub fn rate_seq<F>(mut self, f: &F) -> Population<T, Rated>
        where F: Fn(&T) -> Fitness
    {
        for ind in self.individuals.iter_mut() {
            let fitness = f(&ind.genome);
            ind.fitness = Some(fitness);
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
            ind.fitness = Some(fitness);
        });

        Population {
            individuals: self.individuals,
            _marker: PhantomData,
        }
    }
}

impl<T: Genotype + Debug, R: IsRated> Population<T, R> {
    fn mean_fitness(&self) -> Fitness {
        let sum: Fitness = self.individuals.iter().map(|ind| ind.fitness()).sum();
        sum / Fitness::new(self.len() as f64)
    }
}

impl<T: Genotype + Debug> Population<T, RatedSorted> {

    pub fn into_iter(self) -> ::std::vec::IntoIter<Individual<T>> {
        self.individuals.into_iter()
    }

    pub fn best_individual(&self) -> Option<&Individual<T>> {
        self.individuals.first()
    }

    // Return true if genome at position `i` is fitter that `j`
    //
    // In a sorted population, the individual with the lower index
    // has a better fitness.

    // #[inline]
    // fn is_fitter(&self, i: usize, j: usize) -> bool {
    //    i < j
    // }

    /// Create a single offspring Genome by selecting random parents
    /// from the best `select_size` individuals of the populations.

    fn create_single_offspring<R, M>(&self, select_size: usize, mate: &mut M, rng: &mut R) -> T
        where R: Rng,
              M: Mate<T>
    {
        assert!(select_size > 0 && select_size <= self.len());

        // We do not need tournament selection here as our population is sorted.
        // We simply determine two individuals out of `select_size`.

        let mut parent1 = rng.gen_range(0, select_size);
        let mut parent2 = rng.gen_range(0, select_size);

        // try to find a parent2 != parent1. retry three times.
        for _ in 0..3 {
            if parent2 != parent1 {
                break;
            }
            parent2 = rng.gen_range(0, select_size);
        }

        // `mate` assumes that the first parent performs better.
        if parent1 > parent2 {
            mem::swap(&mut parent1, &mut parent2);
        }

        debug_assert!(parent1 <= parent2);

        mate.mate(&self.individuals[parent1].genome,
                  &self.individuals[parent2].genome,
                  parent1 == parent2,
                  rng)
    }
}


impl<T: Genotype + Debug> Population<T, Rated> {
    pub fn add_individual(&mut self, ind: Individual<T>) {
        self.individuals.push(ind);
    }

    // Return true if genome at position `i` is fitter that `j`
    // fn is_fitter(&self, i: usize, j: usize) -> bool {
    //    self.individuals[i].fitness > self.individuals[j].fitness
    // }

    // higher value of fitness means that the individual is fitter.
    pub fn sort(mut self) -> Population<T, RatedSorted> {
        (&mut self.individuals).sort_by(|a, b| a.fitness().cmp(&b.fitness()).reverse());
        Population {
            individuals: self.individuals,
            _marker: PhantomData,
        }
    }

    pub fn best_individual(&self) -> Option<&Individual<T>> {
        self.individuals.iter().max_by_key(|ind| ind.fitness())
    }

    /// Merge `self` with the first `n` individuals from population `other`.
    pub fn merge(&mut self, other: Population<T, RatedSorted>, n: usize) {
        self.individuals.extend(other.individuals.into_iter().take(n));
    }

    /// Append all individuals of population `other`.
    pub fn append<X: IsRated>(&mut self, other: Population<T, X>) {
        self.individuals.extend(other.individuals.into_iter());
    }

    /// Reproduce a population without niching. Use partition() and `Niches#reproduce()` for
    /// niching.
    ///
    /// Same as `reproduce_into` but returns two Populations (rated, unrated).

    pub fn reproduce<M, R>(self,
                              // The expected size of the new population
                              new_pop_size: f64,
                              // how many of the best individuals of a population are copied as-is into the
                              // new population?
                              elite_percentage: Closed01<f64>,
                              // how many of the best individuals of a populatiion are selected for
                              // reproduction?
                              selection_percentage: Closed01<f64>,
                              mate: &mut M,
                              rng: &mut R) -> (Population<T, Rated>, Population<T, Unrated>)
        where M: Mate<T>,
              R: Rng
    {
        let mut new_unrated_population: Population<T, Unrated> = Population::new();
        let mut new_rated_population: Population<T, Rated> = Population::new();
        self.reproduce_into(new_pop_size, elite_percentage, selection_percentage, mate, &mut new_unrated_population, &mut new_rated_population, rng);

        return (new_rated_population, new_unrated_population);
    }

    /// Reproduce a population without niching. Use partition() and `Niches#reproduce()` for
    /// niching.
    ///
    /// We first sort the population according to it's fitness values.
    /// Then, `selection_percentage` of the best genomes are allowed to mate and produce offspring.
    /// Then, `elite_percentage` of the best genomes is always copied into the new generation.

    fn reproduce_into<M, R>(self,
                              // The expected size of the new population
                              new_pop_size: f64,
                              // how many of the best individuals of a population are copied as-is into the
                              // new population?
                              elite_percentage: Closed01<f64>,
                              // how many of the best individuals of a populatiion are selected for
                              // reproduction?
                              selection_percentage: Closed01<f64>,
                              mate: &mut M,
                              new_unrated_population: &mut Population<T, Unrated>,
                              new_rated_population: &mut Population<T, Rated>,
                              rng: &mut R)
        where M: Mate<T>,
              R: Rng
    {
        // number of elitary individuals to copy from the old generation into the new.
        let elite_size =
            cmp::max(1,
                     probabilistic_round(new_pop_size * elite_percentage.get(), rng) as usize);

        // number of offspring to produce.
        let offspring_size = probabilistic_round(new_pop_size * elite_percentage.inv().get(),
                                                 rng) as usize;

        // number of the best individuals to use for mating.
        let select_size = cmp::min(self.len(), probabilistic_round(new_pop_size *
                                              selection_percentage.get(),
                                              rng) as usize);

        let sorted_pop = self.sort();

        // at first produce `offspring_size` individuals from the top `select_size`
        // individuals.
        if select_size > 0 {
            for _ in 0..offspring_size {
                let offspring = sorted_pop.create_single_offspring(select_size, mate, rng);
                new_unrated_population.add_genome(Box::new(offspring));
            }
        }

        // then copy the elites
        new_rated_population.merge(sorted_pop, elite_size);
    }

    /// Partition the whole population into species (niches)

    pub fn partition<R, C>(self,
                           rng: &mut R,
                           compatibility_threshold: f64,
                           compatibility: &C)
                           -> Niches<T>
        where R: Rng,
              C: Distance<T>
    {
        let mut niches = Niches::new();

        for ind in self.individuals.into_iter() {
            niches.add_individual(ind, compatibility_threshold, compatibility, rng);
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
              G: Fn(usize, &Population<T, Rated>, usize) -> bool
    {
        let mut iteration: usize = 0;
        let mut current_rated_pop = initial_pop.rate_par(self.fitness);
        let mut last_number_of_niches = 1;

        while !goal_condition(iteration, &current_rated_pop, last_number_of_niches) {
            let niches = current_rated_pop.partition(rng, self.compatibility_threshold, self.compatibility);
            last_number_of_niches = niches.num_niches();
            let (new_rated, new_unrated) = niches.reproduce_global(self.pop_size,
                                            self.elite_percentage,
                                            self.selection_percentage,
                                            self.mate,
                                            rng);

            current_rated_pop = new_rated;
            current_rated_pop.append(new_unrated.rate_par(self.fitness));
            iteration += 1;
        }

        return (iteration, current_rated_pop);
    }
}
