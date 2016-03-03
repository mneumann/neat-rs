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

#[derive(Debug)]
struct NicheReproduction {
    old_niche_size: usize,
    new_niche_size: f64,
    percentage_of_population: f64,
    offspring_size: usize,
    select_size: usize,
    elite_size: usize,
}

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
                              top_n_niches: usize,
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
        assert!(top_n_niches <= num_niches);
        let old_niches = {
            let new_niches = self.niches.iter().map(|_| RatedPopulation::new()).collect();
            mem::replace(&mut self.niches, new_niches)
        };

        assert!(old_niches.len() == num_niches);
        assert!(self.niches.len() == num_niches);

        // sort by mean fitness. XXX: use best fitness?
        let mut old_niches_sort: Vec<_> = old_niches.into_iter()
                                                    .map(|niche| {
                                                        (niche.mean_fitness()
                                                              .unwrap_or(Fitness::zero()),
                                                         niche)
                                                    })
                                                    .collect();

        old_niches_sort.sort_by(|a, b| a.0.cmp(&b.0).reverse());

        let total_mean: Fitness = old_niches_sort.iter()
                                                 .take(top_n_niches)
                                                 .map(|&(mean_fitness, _)| mean_fitness)
                                                 .sum();
        println!("total_mean: {:?}", total_mean);

        // Calculate the new size of each niche, which depends on it's mean fitness relative to
        // other niches.

        let new_niche_sizes: Vec<_> = old_niches_sort.iter().enumerate()
                      .map(|(i, &(mean_fitness, ref niche))| {
                          let old_niche_size = niche.len();

                          let percentage_of_population: f64 = if total_mean.get() == 0.0 {
                              // all individuals have a fitness of 0.0.
                              // we will equally allow each niche to procude offspring.
                              1.0 / (num_niches as f64)
                          } else {
                              (mean_fitness / total_mean).get()
                          };

                          // calculate new size of niche
                          assert!(percentage_of_population >= 0.0 &&
                                  percentage_of_population <= 1.0);

                          let new_niche_size = total_pop_size as f64 * percentage_of_population;

                          // number of offspring to produce.

                          let offspring_size = probabilistic_round(new_niche_size *
                                                                   elite_percentage.inv().get(),
                                                                   rng) as usize;

                          // number of elitary individuals to copy from the old generation into the new.
                          let elite_size =
                              cmp::max(1,
                                       probabilistic_round(new_niche_size *
                                                           elite_percentage.get(),
                                                           rng) as usize);


                          // number of the best individuals to use for mating.

                          let select_size =
                              cmp::min(old_niche_size,
                                       probabilistic_round(old_niche_size as f64 *
                                                           selection_percentage.get(),
                                                           rng) as usize);

                          NicheReproduction {
                              old_niche_size: old_niche_size,
                              new_niche_size: new_niche_size,
                              percentage_of_population: percentage_of_population,
                              offspring_size: if i < top_n_niches { offspring_size } else { 0 },
                              select_size: select_size,
                              elite_size: elite_size
                          }
                      })
                  .collect();

        let old_niches: Vec<_> = old_niches_sort.into_iter().map(|(_, niche)| niche).collect();

        println!("niche_sizes: {:?}", new_niche_sizes);

        // XXX: we don't have to sort all niches! only `top_n_niches`
        let ranked_old_niches: Vec<_> = old_niches.into_iter()
                                                  .map(|niche| RankedPopulation::from_rated(niche))
                                                  .collect();

        // Produce offspring. XXX: parallel loop (rng!)

        let mut offspring_population = UnratedPopulation::new(); // XXX: With capacity

        for (niche_id, (ranked_niche, repro)) in ranked_old_niches.iter()
                                                                  .zip(new_niche_sizes.iter())
                                                                  .enumerate() {

            if repro.select_size <= 0 || repro.offspring_size <= 0 {
                continue;
            }

            // produce `offspring_size` individuals from the top `select_size`
            // individuals.

            for _ in 0..repro.offspring_size {
                // let parent1 = rng.gen_range(0, cmp::min(ranked_niche.len(), repro.select_size));

                if rng.gen_range(0, 100) < 50 {
                    // we want to mutate

                    let parent1 = rng.gen_range(0, ranked_niche.len());
                    let parent2 = rng.gen_range(0, ranked_niche.len());

                    // for mutation prefer worse performing genomes

                    let parent = cmp::max(parent1, parent2);

                    let offspring = mate.mate(&ranked_niche.individuals()[parent1].genome(),
                                              &ranked_niche.individuals()[parent2].genome(),
                                              true,
                                              rng);

                    offspring_population.add_unrated_individual(Individual::new_unrated(Box::new(offspring)));
                } else {
                    // crossover

                    // choose the first mating parter.

                    let parent_a1 = rng.gen_range(0, ranked_niche.len());
                    let parent_a2 = rng.gen_range(0, ranked_niche.len());

                    // we prefer better genomes
                    let parent1 = cmp::min(parent_a1, parent_a2);

                    // choose the second mating parter.
                    let parent_b1 = rng.gen_range(0, ranked_niche.len());
                    let parent_b2 = rng.gen_range(0, ranked_niche.len());

                    // prefer the one that has a higher distance.
                    
                    let dist_b1 = compatibility.distance(ranked_niche.individuals()[parent_b1].genome(),
                                                         ranked_niche.individuals()[parent1].genome());


                    let dist_b2 = compatibility.distance(ranked_niche.individuals()[parent_b2].genome(),
                                                         ranked_niche.individuals()[parent1].genome());


                    let parent2 = if dist_b1 > dist_b2 {
                        parent_b1
                        } else { parent_b2 };



                    let offspring = mate.mate(&ranked_niche.individuals()[parent1].genome(),
                                              &ranked_niche.individuals()[parent2].genome(),
                                              parent1 == parent2,
                                              rng);

                    offspring_population.add_unrated_individual(Individual::new_unrated(Box::new(offspring)));
                }


            }
        }

        // keep only the elites in each niche.

        for (niche_id, ranked_niche) in ranked_old_niches.into_iter().enumerate() {
            for ind in ranked_niche.move_individuals()
                                   .into_iter()
                                   .take(new_niche_sizes[niche_id].elite_size) {

                let selected_niche = if rng.gen_range(0, 100) < 10 {
                    rng.gen_range(0, self.niches.len())
                } else {
                    niche_id
                };
                self.niches[selected_niche].add_rated_individual(ind);
            }
        }

        // rate the offspring population

        let rated_offspring_population = RatedPopulation::from_unrated_par(offspring_population,
                                                                           self.fitness_eval);

        // and place it's individuals into the new niches. use random sampling within the niches to
        // determine into which niche to place an individual.

        for ind in rated_offspring_population.move_individuals().into_iter() {

            if rng.gen_range(0, 100) < 20 {
                // randomly insert into a niche.
                let selected_niche = rng.gen_range(0, self.niches.len());
                self.niches[selected_niche].add_rated_individual(ind);
                continue;
            }

            // sort `ind` into the niche that is closest.

            let distances: Vec<_> = self.niches
                                        .iter()
                                        .enumerate()
                                        .map(|(probe_niche_id, probe_niche)| {
                                            let mut distance_sum = 0.0;
                                            for _ in 0..5 {
                                                if let Some(probe_ind) =
                                                       probe_niche.random_individual(rng) {
                                                    distance_sum +=
                                                        compatibility.distance(&probe_ind.genome(),
                                                                               &ind.genome());
                                                }
                                            }
                                            (probe_niche_id, Fitness::new(distance_sum))
                                        })
                                        .collect();

            let selected_niche = distances.iter().min_by_key(|a| a.1).unwrap().0;

            // place it into `selected_niche`

            self.niches[selected_niche].add_rated_individual(ind);
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
