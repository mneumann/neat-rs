use traits::{Genotype, FitnessEval, Distance, Mate};
use population::{UnratedPopulation, RatedPopulation, RankedPopulation, Population, Individual,
                 PopulationWithRating, PopulationWithRank};
use rand::Rng;
use closed01::Closed01;
use fitness::Fitness;
use std::num::Zero;
use std::mem;
use std::cmp;
use prob::probabilistic_round;

pub struct NicheRunner<'a, T, F>
    where T: Genotype + 'a,
          F: FitnessEval<T> + 'a
{
    niches: Vec<RatedPopulation<T>>,
    fitness_eval: &'a F,
    current_iteration: usize,
}

impl<'a, T, F> NicheRunner<'a, T, F>
    where T: Genotype + 'a,
          F: FitnessEval<T> + 'a
{
    pub fn new(fitness_eval: &'a F) -> Self {
        NicheRunner {
            niches: Vec::new(),
            fitness_eval: fitness_eval,
            current_iteration: 0,
        }
    }

    pub fn best_individual(&self) -> Option<&Individual<T>> {
        let mut current_best = None;
        for niche in self.niches.iter() {
            match (current_best, niche.best_individual()) {
                (None, Some(ind)) => {
                    current_best = Some(ind);
                }
                (Some(best), Some(ind)) if ind.fitness() > best.fitness() => {
                    current_best = Some(ind);
                }
                _ => {}
            }
        }
        current_best
    }

    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    pub fn num_niches(&self) -> usize {
        self.niches.len()
    }

    pub fn num_individuals(&self) -> usize {
        self.niches.iter().map(|n| n.len()).sum()
    }

    pub fn has_next_iteration(&mut self, max_iterations: usize) -> bool {
        if self.current_iteration >= max_iterations {
            return false;
        }
        self.current_iteration += 1;
        return true;
    }

    pub fn add_unrated_population_as_niche(&mut self, unrated_pop: UnratedPopulation<T>) {
        let rated_pop = RatedPopulation::from_unrated_par(unrated_pop, self.fitness_eval);
        self.niches.push(rated_pop);
    }

    /// Reproduces offspring within the niches.

    pub fn reproduce<C, M, R>(&mut self,
                              total_pop_size: usize,
                              elite_percentage: Closed01<f64>,
                              selection_percentage: Closed01<f64>,
                              compatibility_threshold: f64,
                              compatibility: &C,
                              mate: &mut M,
                              rng: &mut R)
        where C: Distance<T>,
              M: Mate<T>,
              R: Rng
    {
        assert!(total_pop_size > 0);

        let num_niches = self.niches.len();
        assert!(num_niches > 0);
        let old_niches = {
            let new_niches = self.niches.iter().map(|_| RatedPopulation::new()).collect();
            mem::replace(&mut self.niches, new_niches)
        };

        assert!(old_niches.len() == num_niches);
        assert!(self.niches.len() == num_niches);


        let total_mean: Fitness = old_niches.iter()
                                            .map(|niche| {
                                                niche.mean_fitness().unwrap_or(Fitness::zero())
                                            })
                                            .sum();

        // Calculate the new size of each niche, which depends on it's mean fitness relative to
        // other niches.

        let new_niche_sizes: Vec<_> =
            old_niches.iter()
                      .map(|niche| {
                          let percentage_of_population: f64 = if total_mean.get() == 0.0 {
                              // all individuals have a fitness of 0.0.
                              // we will equally allow each niche to procude offspring.
                              1.0 / (num_niches as f64)
                          } else {
                              (niche.mean_fitness().unwrap_or(Fitness::zero()) / total_mean).get()
                          };

                          // calculate new size of niche
                          assert!(percentage_of_population >= 0.0 &&
                                  percentage_of_population <= 1.0);

                          total_pop_size as f64 * percentage_of_population
                      })
                      .collect();

        let ranked_old_niches: Vec<_> = old_niches.into_iter()
                                                  .map(|niche| RankedPopulation::from_rated(niche))
                                                  .collect();

        // Produce offspring. XXX: parallel loop (rng!)

        for (niche_id, (ranked_niche, &new_niche_size)) in
            ranked_old_niches.iter()
                             .zip(new_niche_sizes.iter())
                             .enumerate() {

            // number of offspring to produce.

            let offspring_size = probabilistic_round(new_niche_size *
                                                     elite_percentage.inv().get(),
                                                     rng) as usize;

            let mut offspring_population = UnratedPopulation::new(); // XXX: With capacity

            // number of the best individuals to use for mating.

            let select_size =
                cmp::min(ranked_niche.len(),
                         probabilistic_round(ranked_niche.len() as f64 *
                                             selection_percentage.get(),
                                             rng) as usize);

            // produce `offspring_size` individuals from the top `select_size`
            // individuals.

            if select_size > 0 {
                for _ in 0..offspring_size {
                    let (parent1, parent2) = ranked_niche.select_parent_indices(select_size,
                                                                                3,
                                                                                rng);
                    debug_assert!(parent1 <= parent2);

                    let offspring = mate.mate(&ranked_niche.individuals()[parent1].genome(),
                                              &ranked_niche.individuals()[parent2].genome(),
                                              parent1 == parent2,
                                              rng);

                    offspring_population.add_unrated_individual(Individual::new_unrated(Box::new(offspring)));
                }
            }

            // now rate the offspring population

            let rated_offspring_population =
                RatedPopulation::from_unrated_par(offspring_population, self.fitness_eval);

            // and place it's individuals into the new niches. use random sampling within the old niches to
            // determine into which niche to place an individual.

            for ind in rated_offspring_population.move_individuals().into_iter() {

                // in case we do not find a niche, use the originating niche

                let mut selected_niche = niche_id;

                'find_niche: for (probe_niche_id, probe_niche) in ranked_old_niches.iter()
                                                                                   .enumerate() {

                    // Is this genome compatible with this niche? Compare `ind` against a random individual
                    // of that `niche`.

                    if let Some(probe_ind) = probe_niche.random_individual(rng) {
                        if compatibility.distance(&probe_ind.genome(), &ind.genome()) <
                           compatibility_threshold {
                            selected_niche = probe_niche_id;
                            break 'find_niche;
                        }
                    }
                }

                // place it into `selected_niche`

                self.niches[selected_niche].add_rated_individual(ind);
            }

        }

        // finally copy the elites into the niches.

        for (niche_id, (ranked_niche, &new_niche_size)) in
            ranked_old_niches.into_iter()
                             .zip(new_niche_sizes.iter())
                             .enumerate() {

            // number of elitary individuals to copy from the old generation into the new.
            let elite_size =
                cmp::max(1,
                         probabilistic_round(new_niche_size *
                                             elite_percentage.get(),
                                             rng) as usize);

            self.niches[niche_id].append_some(ranked_niche, elite_size);
        }
    }

    pub fn into_ranked_population(self) -> RankedPopulation<T> {
        let mut global = RatedPopulation::new();
        for niche in self.niches {
            global.append_all(niche);
        }
        RankedPopulation::from_rated(global)
    }
}
