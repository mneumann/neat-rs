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

        let old_niches: Vec<_> = old_niches_sort.into_iter().map(|(_, niche)| niche).collect();

        // Calculate the new size of each niche, which depends on it's mean fitness relative to
        // other niches.

        let new_niche_sizes: Vec<_> = old_niches.iter().take(top_n_niches)
                                                               .map(|niche| {
                                                                   let old_niche_size = niche.len();

                                                                   let percentage_of_population: f64 = if total_mean.get() == 0.0 {
                                                                       // all individuals have a fitness of 0.0.
                                                                       // we will equally allow each niche to procude offspring.
                                                                       1.0 / (top_n_niches as f64)
                                                                   } else {
                                                                       (niche.mean_fitness().unwrap_or(Fitness::zero()) / total_mean).get()
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
                                                                       offspring_size: offspring_size,
                                                                       select_size: select_size,
                                                                       elite_size: elite_size
                                                                   }
                                                               })
                  .collect();

        println!("niche_sizes: {:?}", new_niche_sizes);

        // XXX: we don't have to sort all niches! only `top_n_niches`
        let ranked_old_niches: Vec<_> = old_niches.into_iter()
                                                  .map(|niche| RankedPopulation::from_rated(niche))
                                                  .collect();

        // Produce offspring. XXX: parallel loop (rng!)

        for (niche_id, (ranked_niche, repro)) in ranked_old_niches.iter()
                                                                  .take(top_n_niches)
                                                                  .zip(new_niche_sizes.iter())
                                                                  .enumerate() {

            let mut offspring_population = UnratedPopulation::new(); // XXX: With capacity

            // produce `offspring_size` individuals from the top `select_size`
            // individuals.

            if repro.select_size > 0 {
                for _ in 0..repro.offspring_size {
                    let (parent1, parent2) = ranked_niche.select_parent_indices(repro.select_size,
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

                // sort `ind` into the niche that is closest.

                let distances: Vec<_> = ranked_old_niches.iter().enumerate().map(|(probe_niche_id, probe_niche)| {
                    let mut distance_sum = 0.0;
                    for _ in 0..10 {
                        if let Some(probe_ind) = probe_niche.random_individual(rng) {
                            distance_sum += compatibility.distance(&probe_ind.genome(), &ind.genome());
                        }
                    }
                    (probe_niche_id, Fitness::new(distance_sum))
                }).collect();


                let selected_niche = distances.iter().min_by_key(|a| a.1).unwrap().0;

                /*
                 
                // in case we do not find a niche, use the originating niche

                let mut selected_niche = niche_id;

                // Sample `2 * number of niches` times a randomly choosen niche.

                'select_niche: for _ in 0..(2 * ranked_old_niches.len()) {
                    let probe_niche_id = rng.gen_range(0, ranked_old_niches.len());

                    let probe_niche = &ranked_old_niches[probe_niche_id];

                    // Is this genome compatible with this niche? Compare `ind` against a random individual
                    // of that `niche`.

                    if let Some(probe_ind) = probe_niche.random_individual(rng) {
                        if compatibility.distance(&probe_ind.genome(), &ind.genome()) <
                           compatibility_threshold {
                            selected_niche = probe_niche_id;
                            break 'select_niche;
                        }
                    }
                }

                */


                // place it into `selected_niche`

                self.niches[selected_niche].add_rated_individual(ind);
            }

        }

        // finally copy the elites into the niches.

        //for (niche_id, (ranked_niche, repro)) in ranked_old_niches.into_iter()
        for (niche_id, ranked_niche) in ranked_old_niches.into_iter()
                                                                  //.zip(new_niche_sizes.iter())
                                                                  .enumerate() {

            if niche_id < top_n_niches {
                self.niches[niche_id].append_some(ranked_niche, new_niche_sizes[niche_id].elite_size);
            } else {
                // copy the niche
                self.niches[niche_id].append_all(ranked_niche);
            }
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
