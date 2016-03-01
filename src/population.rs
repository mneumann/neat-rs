use fitness::Fitness;
use traits::{Genotype, Distance, Mate, FitnessEval};
use prob::probabilistic_round;
use distribute::DistributeInterval;

use rand::Rng;
use closed01::Closed01;
use std::marker::PhantomData;
use std::fmt::Debug;
use std::cmp;
use std::mem;
use std::f64::{INFINITY, NEG_INFINITY};
use rayon::par_iter::*;

#[derive(Debug)]
pub struct Individual<T: Debug + Genotype> {
    fitness: Option<Fitness>,
    genome: Box<T>,
}

impl<T: Debug + Genotype> Individual<T> {
    pub fn has_fitness(&self) -> bool {
        self.fitness.is_some()
    }

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
struct NicheFitnessStat {
    pub min_fitness: Fitness,
    pub max_fitness: Fitness,
    pub mean_fitness: Fitness,
}

#[derive(Debug)]
pub struct Niche<T: Genotype + Debug> {
    population: Population<T, Rated>,

    centroid: Option<usize>,

    // Stores a log of NicheFitnessStat entries over time. An entry is added each time
    // `log_fitness()` is called.
    fitness_log: Vec<NicheFitnessStat>,
}


impl<T: Genotype + Debug> Niche<T> {
    fn from_population(pop: Population<T, Rated>) -> Self {
        assert!(pop.len() > 0);
        let best_index = pop.best_individual_index();
        Niche {
            population: pop,
            centroid: best_index,
            fitness_log: Vec::new(),
        }
    }

    /// Logs the fitness statistics

    fn log_fitness(&mut self) {
        let stat = NicheFitnessStat {
            min_fitness: self.min_fitness(),
            max_fitness: self.max_fitness(),
            mean_fitness: self.mean_fitness()
        };
        self.fitness_log.push(stat);
    }

    fn fitness_log(&self) -> &[NicheFitnessStat] {
        &self.fitness_log
    }

    pub fn fitness_improvement<F>(&self, timesteps: usize, mut select_fitness: F) -> Option<f64>
        where F: Fn(&NicheFitnessStat) -> Fitness
    {
        if timesteps > self.fitness_log.len() {
            None
        } else {
            let last_fitness = select_fitness(&self.fitness_log[self.fitness_log.len() - 1]).get();
            let ago_fitness = select_fitness(&self.fitness_log[self.fitness_log.len() - timesteps]).get();

            let improvement = last_fitness - ago_fitness;
            Some(improvement)
        }
    }


    /// Calculate the new size of this niche. Depends on the `total_mean` fitness of all niches
    /// and this niche's own mean_fitness.
    ///
    /// `num_niches`: Total number of niches.
    /// `global_pop_size`: Size of the global population.

    pub fn determine_new_niche_size(&self,
                                    total_mean: Fitness,
                                    num_niches: usize,
                                    global_pop_size: usize)
                                    -> f64 {
        assert!(num_niches > 0);
        assert!(global_pop_size > 0);

        let percentage_of_population: f64 = if total_mean.get() == 0.0 {
            // all individuals have a fitness of 0.0.
            // we will equally allow each niche to procude offspring.
            1.0 / (num_niches as f64)
        } else {
            (self.mean_fitness() / total_mean).get()
        };

        // calculate new size of niche
        assert!(percentage_of_population >= 0.0 && percentage_of_population <= 1.0);

        let new_niche_size = global_pop_size as f64 * percentage_of_population;

        return new_niche_size;
    }

    /// Reproduces a niche locally

    pub fn reproduce_locally<M, R, F>(&mut self,
                                      new_niche_size: f64,
                                      elite_percentage: Closed01<f64>,
                                      selection_percentage: Closed01<f64>,
                                      mate: &mut M,
                                      fitness_eval: &F,
                                      rng: &mut R)
        where M: Mate<T>,
              R: Rng,
              F: FitnessEval<T>
    {
        let old_population = mem::replace(&mut self.population, Population::new());
        let mut new_unrated_population: Population<T, Unrated> = Population::new();

        old_population.reproduce_into(new_niche_size,
                                      elite_percentage,
                                      selection_percentage,
                                      mate,
                                      &mut new_unrated_population,
                                      &mut self.population,
                                      rng);

        self.population.append(new_unrated_population.rate_par(fitness_eval));
    }


    pub fn len(&self) -> usize {
        self.population.len()
    }

    pub fn best_individual(&self) -> &Individual<T> {
        self.population.best_individual().unwrap()
    }

    fn mean_fitness(&self) -> Fitness {
        self.population.mean_fitness()
    }

    fn max_fitness(&self) -> Fitness {
        self.population.max_fitness()
    }

    fn min_fitness(&self) -> Fitness {
        self.population.min_fitness()
    }



    /// Returns a reference to the centroid element or a random element of the niche.

    fn centroid_or_random_individual<R: Rng>(&self, rng: &mut R) -> &Individual<T> {
        self.centroid.and_then(|i| self.population.individuals.get(i)).unwrap_or_else(|| self.population.random_individual(rng))
    }

    /// Returns a reference to a random element of the niche.

    fn random_individual<R: Rng>(&self, rng: &mut R) -> &Individual<T> {
        self.population.random_individual(rng)
    }

    fn add_individual(&mut self, ind: Individual<T>) {
        assert!(ind.has_fitness());
        self.population.add_individual(ind);
    }

    fn from_individual(ind: Individual<T>) -> Self {
        assert!(ind.has_fitness());

        let mut pop = Population::new();
        pop.add_individual(ind);
        Niche::from_population(pop)
    }
}

#[derive(Debug)]
pub struct Niches<T: Genotype + Debug> {
    niches: Vec<Niche<T>>,
}

impl<T: Genotype + Debug> Niches<T> {
    pub fn new() -> Self {
        Niches { niches: Vec::new() }
    }


    /// Returns a reference to two (distinct) niches.

    fn two_random_distinct_niches<R: Rng>(&self, rng: &mut R) -> (&Niche<T>, &Niche<T>) {
        let (a, b) = match self.niches.len() {
            0 => panic!(),
            1 => (0, 0),
            2 => (0, 1),
            len => {
                let a = rng.gen_range(0, len);
                let mut b = rng.gen_range(0, len);
                for _ in 0..3 {
                    if b != a {
                        break;
                    }
                    b = rng.gen_range(0, len);
                }
                (a, b)
            }
        };

        (&self.niches[a], &self.niches[b])
    }



    /// Insert the individuals of `population` into the first matching niche, according to the
    /// `compatibility` and `compatibility_threshold`.
    ///
    /// This will not create a new niche if `max_num_niches` is reached.

    pub fn insert_population_threshold<C, R, RA>(&mut self,
                                                 population: Population<T, RA>,
                                                 compatibility_threshold: f64,
                                                 compatibility: &C,
                                                 max_num_niches: usize,
                                                 rng: &mut R)
        where C: Distance<T>,
              R: Rng,
              RA: IsRated
    {
        assert!(max_num_niches > 0);
        for ind in population.individuals.into_iter() {
            if let Some(niche) = self.find_first_matching_niche(&ind,
                                                                compatibility_threshold,
                                                                compatibility,
                                                                rng) {
                niche.add_individual(ind);
                continue;
            }

            if self.num_niches() < max_num_niches {
                // if no compatible niche was found, create a new niche containing this individual.
                self.add_niche(Niche::from_individual(ind));
            } else {
                // the maximum number of niches has been reached. insert into the best matching
                // niche.
                self.find_best_matching_niche(&ind, compatibility, rng).unwrap().add_individual(ind);
            }
        }
    }


    pub fn best_individual(&self) -> &Individual<T> {
        assert!(self.niches.len() > 0);
        let mut best = self.niches[0].best_individual();
        for i in 1..self.niches.len() {
            let best2 = self.niches[i].best_individual();
            if best2.fitness() > best.fitness() {
                best = best2;
            }
        }
        return best;
    }


    /// Creates a `Niches` with a single niche containing the whole `Population`.

    pub fn from_single_population(pop: Population<T, Rated>) -> Self {
        let total = pop.len();
        let niche = Niche::from_population(pop);
        Niches { niches: vec![niche] }
    }

    /// Collapse all niches into a single `Population`.

    pub fn collapse(self) -> Population<T, Rated> {
        assert!(!self.niches.is_empty());
        let tot = self.num_individuals();

        let mut iter = self.niches.into_iter();
        let mut pop = iter.next().unwrap().population;

        for niche in iter {
            pop.append(niche.population);
        }

        assert!(tot == pop.len());
        pop
    }

    /// The sum of all "mean fitnesses" of all niches.

    fn total_mean(&self) -> Fitness {
        self.niches.iter().map(|niche| niche.population.mean_fitness()).sum()
    }

    /// Calculates the total number of individuals of all niches.

    pub fn num_individuals(&self) -> usize {
        self.niches.iter().map(|niche| niche.len()).sum()
    }

    /// Number of niches

    pub fn num_niches(&self) -> usize {
        self.niches.len()
    }

    /// Add a new niche to the `Niches`.

    pub fn add_niche(&mut self, niche: Niche<T>) {
        assert!(niche.len() > 0);
        self.niches.push(niche);
    }

    /// Add an individual to the first matching niche (given by the `compatibility_threshold` and
    /// `compatibility` function, comparing against a random individual of that niche.
    /// If no niche matches, create a new.

    pub fn find_first_matching_niche<'a, R, C>(&'a mut self,
                                               ind: &Individual<T>,
                                               compatibility_threshold: f64,
                                               compatibility: &C,
                                               rng: &mut R)
                                               -> Option<&'a mut Niche<T>>
        where R: Rng,
              C: Distance<T>
    {
        for niche in self.niches.iter_mut() {

            // Is this genome compatible with this niche? Compare `ind` against a random individual
            // of that `niche`.

            if compatibility.distance(&niche.centroid_or_random_individual(rng).genome, &ind.genome) <
               compatibility_threshold {
                return Some(niche);
            }
        }
        return None;
    }

    pub fn find_best_matching_niche<'a, R, C>(&'a mut self,
                                               ind: &Individual<T>,
                                               compatibility: &C,
                                               rng: &mut R)
                                               -> Option<&'a mut Niche<T>>
        where R: Rng,
              C: Distance<T>
    {
        assert!(self.niches.len() > 0);

        let mut best_niche = None;

        for (i, niche) in self.niches.iter().enumerate() {
            let dist = compatibility.distance(&niche.random_individual(rng).genome, &ind.genome);

            best_niche = match best_niche {
                None => {
                    Some((i, dist))
                }
                Some((_j, best_dist)) if dist < best_dist => {
                   Some((i, dist))
                }
                Some((j, best_dist)) => {
                    Some((j, best_dist))
                }
            };
        }

        if let Some((i, _)) = best_niche {
            Some(&mut self.niches[i])
        } else {
            None
        }
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

        for niche in self.niches.into_iter() {
            let niche_size = niche.determine_new_niche_size(total_mean, num_niches, new_pop_size);
            niche.population.reproduce_into(niche_size,
                                            elite_percentage,
                                            selection_percentage,
                                            mate,
                                            &mut new_unrated_population,
                                            &mut new_rated_population,
                                            rng);
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
        where F: FitnessEval<T>
    {
        for ind in self.individuals.iter_mut() {
            let fitness = f.fitness(&ind.genome);
            ind.fitness = Some(fitness);
        }
        Population {
            individuals: self.individuals,
            _marker: PhantomData,
        }
    }

    pub fn rate_par<F>(mut self, f: &F) -> Population<T, Rated>
        where F: FitnessEval<T>
    {
        self.individuals.par_iter_mut().for_each(|ind| {
            let fitness = f.fitness(&ind.genome);
            ind.fitness = Some(fitness);
        });

        Population {
            individuals: self.individuals,
            _marker: PhantomData,
        }
    }
}

impl<T: Genotype + Debug> Into<Population<T, Rated>> for Population<T, RatedSorted> {
    fn into(self) -> Population<T, Rated> {
        Population {
            individuals: self.individuals,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct SampleCompatibilityDistance {
    pub min: f64,
    pub max: f64,
    pub sum: f64,
    pub samples: usize,
}

impl SampleCompatibilityDistance {
    pub fn new() -> Self {
        SampleCompatibilityDistance {
            min: INFINITY,
            max: NEG_INFINITY,
            sum: 0.0,
            samples: 0,
        }
    }

    pub fn span(&self) -> Option<f64> {
        if self.samples > 0 {
            Some(self.max - self.min)
        } else {
            None
        }
    }

    pub fn mean(&self) -> Option<f64> {
        if self.samples > 0 {
            Some(self.sum / (self.samples as f64))
        } else {
            None
        }
    }

    pub fn add_sample(&mut self, distance: f64) {
        if distance < self.min {
            self.min = distance;
        }
        if distance > self.max {
            self.max = distance;
        }
        self.sum += distance;
        self.samples += 1;
    }

    pub fn add(&mut self, other: &Self) {
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.sum += other.sum;
        self.samples += other.samples;
    }
}

impl<T: Genotype + Debug, RA: IsRated> Population<T, RA> {
    fn mean_fitness(&self) -> Fitness {
        let sum: Fitness = self.individuals.iter().map(|ind| ind.fitness()).sum();
        sum / Fitness::new(self.len() as f64)
    }

    /// Returns a reference to a random element of the population.

    fn random_individual<R: Rng>(&self, rng: &mut R) -> &Individual<T> {
        rng.choose(&self.individuals).unwrap()
    }

    /// Returns a reference to two distinct random elements of the population.

    fn two_random_distinct_individuals<R: Rng>(&self, rng: &mut R) -> Option<(&Individual<T>, &Individual<T>)> {
        if self.len() >= 2 {
            let a = rng.gen_range(0, self.len());
            for _ in 0..3 {
                let b = rng.gen_range(0, self.len());
                if b != a {
                    return Some((&self.individuals[a], &self.individuals[b]));
                }
            }
        }
        None
    }

    /// Samples `n_samples` times the distance between two randomly choosen individuals of
    /// the population and determines the minimum, maximum and sum of the distance.

    fn sample_compatibility_distance<C, R>(&self,
                                           n_samples: usize,
                                           compatibility: &C,
                                           sample_distance: &mut SampleCompatibilityDistance,
                                           rng: &mut R)
        where C: Distance<T>,
              R: Rng
    {
        for _ in 0..n_samples {
            if let Some((a, b)) = self.two_random_distinct_individuals(rng) { 
                let distance = compatibility.distance(&a.genome, &b.genome);
                sample_distance.add_sample(distance);
            }
        }
    }

    /// Partition the whole population into species (niches)

    pub fn partition<C, R>(self,
                           compatibility_threshold: f64,
                           compatibility: &C,
                           max_num_niches: usize,
                           rng: &mut R)
                           -> Niches<T>
        where C: Distance<T>,
              R: Rng
    {
        let mut niches = Niches::new();
        niches.insert_population_threshold(self, compatibility_threshold, compatibility, max_num_niches, rng);
        niches
    }

    /// Partition the population into `n` niches.
    ///
    /// * (Sort population).
    /// * Split population into `n` regions.
    /// * Within each region, select a random individual as `centroid`.
    /// * This becomes the centroid of that a niche.
    /// * Now place each remaining individual into the niche that has the closest distance.

    pub fn partition_n<C, R>(self, n: usize, compatibility: &C, rng: &mut R) -> Niches<T>
        where C: Distance<T>,
              R: Rng,
              Self: Into<Population<T, Rated>>
    {
        assert!(n > 0);
        assert!(self.len() > 0);

        if n == 1 {
            return Niches::from_single_population(self.into());
        }

        if n > self.len() {
            warn!("Number of niches ({}) > number of individuals ({}). Cap!",
                  n,
                  self.len());
        }
        let n = cmp::min(n, self.len());

        assert!(n > 1 && n <= self.len());

        let max_size_per_niche = cmp::max(2, (self.len() * 2) / n);

        // We distribute `n` points within the internal (0..len).
        // Then around each point, we select a random individual as centroid.
        let distribute = DistributeInterval::new(n, 0.0, self.len() as f64);

        // We select individuals within [pt - half_width .. pt + half_width].
        // It's okay to have overlap.
        let half_width = ((self.len() as f64 / n as f64) / 2.0) + 1.0;

        // For each of the `n` niches, determine a centroid individual (index)
        let niche_centroids: Vec<usize> =
            distribute.map(|pt| {
                          // find a centroid for niche by looking around `pt`
                          let centroid_idx_f = rng.gen_range(pt - half_width, pt + half_width)
                                                  .max(0.0);
                          let centroid_idx: usize =
                              cmp::min(self.len() - 1,
                                       probabilistic_round(centroid_idx_f, rng) as usize);
                          assert!(centroid_idx < self.len());
                          centroid_idx
                      })
                      .collect();

        assert!(niche_centroids.len() == n);

        // record the size (number of individuals) within each niche.
        let mut niche_sizes: Vec<usize> = niche_centroids.iter().map(|_| 0).collect();

        // in rare cases, two adjacent niches could have the same centroid.

        // For each individual `ind` test all centroids. Sort it into that niche,
        // whoose centroid is closest. If two niche centroids have equal distance to `ind`,
        // take the smaller niche.

        let niche_assignment: Vec<usize> =
            self.individuals
                .iter()
                .map(|ind| {
                    let mut best_niche = None;

                    // test all other niches
                    for i in 0..n {

                        // Only test niche if it isn't full yet!
                        if niche_sizes[i] > max_size_per_niche {
                            continue;
                        }

                        let dist = compatibility.distance(&self.individuals[niche_centroids[i]]
                                                               .genome,
                                                          &ind.genome);

                        best_niche = match best_niche {
                            None => Some((i, dist)),
                            Some((best_niche_i, best_dist)) => {
                                if (dist < best_dist) ||
                                   ((dist == best_dist) &&
                                    niche_sizes[i] < niche_sizes[best_niche_i]) {
                                    Some((i, dist))
                                } else {
                                    // keep old best niche
                                    Some((best_niche_i, best_dist))
                                }
                            }
                        };
                    }

                    let best_niche_i = best_niche.unwrap().0;

                    // sort `ind` into niche `best_niche`. Increase size of that niche.
                    niche_sizes[best_niche_i] += 1;
                    best_niche_i
                })
                .collect();

        assert!(niche_assignment.len() == self.individuals.len());

        // Create the niches.
        let mut niche_pops: Vec<Population<T, Rated>> =
            niche_centroids.iter()
                           .map(|_| Population::<T, Rated>::new())
                           .collect();

        // And put each individual into it's niche.
        for (ind, niche_id) in self.individuals.into_iter().zip(niche_assignment) {
            niche_pops[niche_id].add_individual(ind);
        }

        let mut niches = Niches::new();

        for niche_pop in niche_pops.into_iter() {
            if niche_pop.len() > 0 {
                // we reject empty niches.
                niches.add_niche(Niche::from_population(niche_pop));
            }
        }

        assert!(niches.num_niches() <= n);

        if niches.num_niches() != n {
            warn!("Niche number mismatch. Expected: {}. Actual: {}",
                  n,
                  niches.num_niches());
        }

        niches
    }
}

impl<T: Genotype + Debug> Population<T, RatedSorted> {
    pub fn into_iter(self) -> ::std::vec::IntoIter<Individual<T>> {
        self.individuals.into_iter()
    }

    fn min_fitness(&self) -> Fitness {
        self.worst_individual().unwrap().fitness()
    }

    fn max_fitness(&self) -> Fitness {
        self.best_individual().unwrap().fitness()
    }


    pub fn best_individual_index(&self) -> Option<usize> {
        if self.individuals.len() > 0 {
            Some(0)
        } else {
            None
        }
    }

    pub fn worst_individual(&self) -> Option<&Individual<T>> {
        self.individuals.last()
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

    fn min_fitness(&self) -> Fitness {
        self.worst_individual().unwrap().fitness()
    }

    fn max_fitness(&self) -> Fitness {
        self.best_individual().unwrap().fitness()
    }

    pub fn best_individual_index(&self) -> Option<usize> {
        (0..self.individuals.len()).into_iter().max_by_key(|&i| self.individuals[i].fitness())
    }

    pub fn best_individual(&self) -> Option<&Individual<T>> {
        self.individuals.iter().max_by_key(|ind| ind.fitness())
    }

    pub fn worst_individual(&self) -> Option<&Individual<T>> {
        self.individuals.iter().min_by_key(|ind| ind.fitness())
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
                           rng: &mut R)
                           -> (Population<T, Rated>, Population<T, Unrated>)
        where M: Mate<T>,
              R: Rng
    {
        let mut new_unrated_population: Population<T, Unrated> = Population::new();
        let mut new_rated_population: Population<T, Rated> = Population::new();
        self.reproduce_into(new_pop_size,
                            elite_percentage,
                            selection_percentage,
                            mate,
                            &mut new_unrated_population,
                            &mut new_rated_population,
                            rng);

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
        let select_size =
            cmp::min(self.len(),
                     probabilistic_round(new_pop_size * selection_percentage.get(), rng) as usize);

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
}

pub struct NicheRunner<'a, T, F>
    where T: Genotype + Debug + 'a,
          F: FitnessEval<T> + 'a
{
    niches: Niches<T>,
    fitness: &'a F,
    current_iteration: usize,
}

impl<'a, T, F> NicheRunner<'a, T, F>
    where T: Genotype + Debug + 'a,
          F: FitnessEval<T> + 'a
{
    pub fn new(fitness: &'a F) -> Self {
        NicheRunner {
            niches: Niches::new(),
            fitness: fitness,
            current_iteration: 0,
        }
    }

    pub fn best_individual(&self) -> &Individual<T> {
        assert!(self.niches.num_niches() > 0);
        self.niches.best_individual()
    }

    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    pub fn num_niches(&self) -> usize {
        self.niches.num_niches()
    }

    pub fn num_individuals(&self) -> usize {
        self.niches.num_individuals()
    }

    pub fn has_next_iteration(&mut self, max_iterations: usize) -> bool {
        if self.current_iteration >= max_iterations {
            return false;
        }
        self.current_iteration += 1;
        return true;
    }

    pub fn add_unrated_population_as_niche(&mut self, pop: Population<T, Unrated>) {
        self.niches.add_niche(Niche::from_population(pop.rate_par(self.fitness)));
    }

    pub fn partition_threshold<C, R>(&mut self,
                                     compatibility_threshold: f64,
                                     compatibility: &C,
                                     max_num_niches: usize,
                                     rng: &mut R)
        where C: Distance<T>,
              R: Rng
    {
        let niches = mem::replace(&mut self.niches, Niches::new());
        self.niches = niches.collapse().partition(compatibility_threshold, compatibility, max_num_niches, rng);
    }

    pub fn partition_n_sorted<C, R>(&mut self, n: usize, compatibility: &C, rng: &mut R)
        where C: Distance<T>,
              R: Rng
    {
        let niches = mem::replace(&mut self.niches, Niches::new());
        self.niches = niches.collapse().sort().partition_n(n, compatibility, rng);
    }

    /// If a niche does not show a signification improvement > `improvement_threshold` within the last `timesteps`
    /// redistribute it's individuals to the remaining niches.
    ///
    /// TODO: Niches which do not have a minimum number of elements will be destroyed. 

    pub fn redistribute_niches_with_no_improvement<C, R>(&mut self,
                                                         improvement_threshold: f64,
                                                         timesteps: usize,
                                                         max_num_niches: usize,
                                                         compatibility_threshold: f64,
                                                         compatibility: &C,
                                                         rng: &mut R)
                                                         -> usize
        where C: Distance<T>,
              R: Rng
    {
        assert!(timesteps > 0);
        assert!(max_num_niches > 0);

        // record mean_fitness for all niches.

        {
            for niche in self.niches.niches.iter_mut() {
                niche.log_fitness();
            }
        }

        let old_niches = mem::replace(&mut self.niches, Niches::new());
        let mut niches_to_redistribute = Vec::new();

        for niche in old_niches.niches.into_iter() {
            match niche.fitness_improvement(timesteps, |f| f.max_fitness) {
                Some(niche_improvement) if niche_improvement < improvement_threshold => {
                    // this niche did not improve good enough
                    niches_to_redistribute.push(niche);
                }
                _ => {
                    // keep it as is
                    self.niches.add_niche(niche);
                }
            }
        }

        let redistributes = niches_to_redistribute.len();
        for niche in niches_to_redistribute {
            self.niches.insert_population_threshold(niche.population,
                                                    compatibility_threshold,
                                                    compatibility,
                                                    max_num_niches,
                                                    rng);
        }
        redistributes
    }

    pub fn sample_compatibility_distance<C, R>(&self,
                                               n_samples_per_niche: usize,
                                               compatibility: &C,
                                               rng: &mut R)
                                               -> SampleCompatibilityDistance
        where C: Distance<T>,
              R: Rng
    {
        let mut sample_distance = SampleCompatibilityDistance::new();
        for niche in self.niches.niches.iter() {
            niche.population.sample_compatibility_distance(n_samples_per_niche, compatibility, &mut sample_distance, rng);
        }
        sample_distance
    }

    pub fn inter_niche_compatibility_distance<C, R>(&self,
                                               samples: usize,
                                               compatibility: &C,
                                               rng: &mut R)
                                               -> Vec<f64>
        where C: Distance<T>,
              R: Rng
    {
        assert!(self.niches.num_niches() > 0);

        (0..samples).map(|_| {
            let (niche_a, niche_b) = self.niches.two_random_distinct_niches(rng);
            let a = niche_a.random_individual(rng);
            let b = niche_b.random_individual(rng);

            compatibility.distance(&a.genome, &b.genome)
        }).collect()
    }

    /// Reproduces offspring within the niches. The niches are not destroyed.

    pub fn reproduce_niche_locally<M, R>(&mut self,
                                         new_total_pop_size: usize,
                                         elite_percentage: Closed01<f64>,
                                         selection_percentage: Closed01<f64>,
                                         mate: &mut M,
                                         rng: &mut R)
        where M: Mate<T>,
              R: Rng
    {
        let total_mean = self.niches.total_mean();
        let num_niches = self.niches.num_niches();

        // XXX: do in parallel
        for niche in self.niches.niches.iter_mut() {
            let new_niche_size = niche.determine_new_niche_size(total_mean,
                                                                num_niches,
                                                                new_total_pop_size);
            niche.reproduce_locally(new_niche_size,
                                    elite_percentage,
                                    selection_percentage,
                                    mate,
                                    self.fitness,
                                    rng);
        }
    }

    pub fn reproduce_global<M, R>(&mut self,
                                  new_pop_size: usize,
                                  elite_percentage: Closed01<f64>,
                                  selection_percentage: Closed01<f64>,
                                  mate: &mut M,
                                  rng: &mut R)
        where M: Mate<T>,
              R: Rng
    {
        let niches = mem::replace(&mut self.niches, Niches::new());
        let (mut new_rated, new_unrated) = niches.reproduce_global(new_pop_size,
                                                                   elite_percentage,
                                                                   selection_percentage,
                                                                   mate,
                                                                   rng);

        new_rated.append(new_unrated.rate_par(self.fitness));
        self.niches = Niches::from_single_population(new_rated);
    }

    pub fn into_population(self) -> Population<T, Rated> {
        self.niches.collapse()
    }
}

pub struct Runner<'a, T, C, M, F>
    where T: Genotype + Debug,
          C: Distance<T> + 'a,
          M: Mate<T> + 'a,
          F: FitnessEval<T> + 'a
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
          F: FitnessEval<T> + 'a
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
            let niches = current_rated_pop.sort().partition_n(5, self.compatibility, rng);
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
