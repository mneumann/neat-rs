#![feature(iter_arith)]

extern crate rand;

use std::cmp;
use innovation::{Innovation, InnovationContainer};
use alignment::Alignment;
use rand::{Rng, Closed01};

mod innovation;
mod selection;
mod alignment;
mod compatibility;

pub trait Gene: Clone {
    fn weight_distance(&self, other: &Self) -> f64;
}

#[derive(Copy, Clone)]
enum NodeType {
    Input,
    Output,
    Hidden,
}

#[derive(Clone)]
struct NodeGene {
    node_type: NodeType,
}

// To avoid node collisions we use Innovation numbers instead of node
// ids.
#[derive(Clone)]
struct LinkGene {
    // This points to the NodeGene of that innovation
    source_node_gene: Innovation,
    // This points to the NodeGene of that innovation
    target_node_gene: Innovation,
    weight: f64,
    active: bool,
}

impl Gene for LinkGene {
    fn weight_distance(&self, other: &LinkGene) -> f64 {
        self.weight - other.weight
    }
}

#[derive(Clone)]
struct Genome {
    link_genes: InnovationContainer<LinkGene>,
    node_genes: InnovationContainer<NodeGene>,
}


/// These describe the probabilities to take a gene from one of the 
/// parents during crossover.
/// XXX: It's probably faster to not use floats here.
struct CrossoverProbabilities {
    /// Probability to take a matching gene from the fitter (first) parent.
    prob_match1: Closed01<f32>,

    /// Probability to take a disjoint gene from the fitter (first) parent.
    prob_disjoint1: Closed01<f32>,

    /// Probability to take an excess gene from the fitter (first) parent.
    prob_excess1: Closed01<f32>,

    /// Probability to take a disjoint gene from the less fit parent.
    prob_disjoint2: Closed01<f32>,

    /// Probability to take an excess gene from the less fit parent.
    prob_excess2: Closed01<f32>,
}

fn is_probable<R: Rng>(prob: &Closed01<f32>, rng: &mut R) -> bool {
    if prob.0 < 1.0 {
        let v: f32 = rng.gen(); // half open [0, 1)
        debug_assert!(v >= 0.0 && v < 1.0);
        v < prob.0
    } else {
        true
    }
}

/// We assume `parent1` to be the fitter parent. Takes gene 
/// either from `parent1` or `parent2` according to
/// the probabilities specified and the relative fitness of the parents.
fn crossover<T: Clone, R: Rng>(parent_left: &InnovationContainer<T>,
                               parent_right: &InnovationContainer<T>,
                               p: &CrossoverProbabilities,
                               rng: &mut R)
                               -> InnovationContainer<T> {

    let mut offspring = InnovationContainer::new();

    parent_left.align(parent_right,
                      &mut |innov, alignment| {
                          match alignment {
                              Alignment::Match(gene_left, gene_right) => {
                                  if is_probable(&p.prob_match1, rng) {
                                      offspring.insert(innov, gene_left.clone());
                                  } else {
                                      offspring.insert(innov, gene_right.clone());
                                  }
                              }

                              Alignment::DisjointLeft(gene_left) => {
                                  if is_probable(&p.prob_disjoint1, rng) {
                                      offspring.insert(innov, gene_left.clone());
                                  }
                              }

                              Alignment::DisjointRight(gene_right) => {
                                  if is_probable(&p.prob_disjoint2, rng) {
                                      offspring.insert(innov, gene_right.clone());
                                  }
                              }

                              Alignment::ExcessLeft(gene_left) => {
                                  if is_probable(&p.prob_excess1, rng) {
                                      offspring.insert(innov, gene_left.clone());
                                  }
                              }

                              Alignment::ExcessRight(gene_right) => {
                                  if is_probable(&p.prob_excess2, rng) {
                                      offspring.insert(innov, gene_right.clone());
                                  }
                              }
                          }
                      });

    offspring
}

type Fitness = f64;

// A niche can never be empty!
struct Niche {
    genomes: Vec<(Fitness, Box<Genome>)>,
    fitness_sum: Fitness,
}

impl Niche {
    fn len(&self) -> usize {
        let l = self.genomes.len();
        assert!(l > 0);
        l
    }

    // Return true if genome at position `i` is better that `j`
    fn compare_ij(&self, i: usize, j: usize) -> bool {
        self.genomes[i].0 > self.genomes[j].0
    }

    fn new_with(fitness: Fitness, genome: Box<Genome>) -> Niche {
        assert!(fitness >= 0.0);
        Niche {
            genomes: vec![(fitness, genome)],
            fitness_sum: fitness,
        }
    }

    fn mean_fitness(&self) -> Fitness {
        self.fitness_sum / (self.len() as Fitness)
    }

    fn add(&mut self, fitness: Fitness, genome: Box<Genome>) {
        assert!(fitness >= 0.0);
        self.genomes.push((fitness, genome));
        self.fitness_sum += fitness;
    }

    fn sort(mut self) -> Self {
        (&mut self.genomes).sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(cmp::Ordering::Less));
        self
    }
}

struct RatedPopulation {
    genomes: Vec<(Fitness, Box<Genome>)>,
}

struct UnratedPopulation {
    genomes: Vec<Box<Genome>>,
}

impl RatedPopulation {
    fn len(&self) -> usize {
        self.genomes.len()
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
    fn produce_offspring<R>(self,
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
                            coeff: &compatibility::CompatibilityCoefficients)
                            -> (RatedPopulation, UnratedPopulation)
        where R: Rng
    {
        assert!(elite_percentage.0 <= selection_percentage.0); // XXX
        assert!(self.len() > 0);
        let niches = self.partition(rng, threshold, coeff);
        assert!(niches.len() > 0);
        let total_mean: Fitness = niches.iter().map(|n| n.mean_fitness()).sum();
        // XXX: total_mean = 0.0?

        let mut new_unrated_population = Vec::new();
        let mut new_rated_population = Vec::new();

        for mut niche in niches.into_iter() {
            // calculate new size of niche, and size of elites, and selection size.
            let percentage_of_population: f64 = niche.mean_fitness() / total_mean;
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
                    let offspring = sorted_niche.genomes[parent1].1.clone();
                    // XXX: mate(parent1, parent2)
                    new_unrated_population.push(offspring);
                    n -= 1;
                }
            }

            // now copy the elites
            new_rated_population.extend(sorted_niche.genomes.into_iter().take(elite_size));
        }


        (RatedPopulation { genomes: new_rated_population },
         UnratedPopulation { genomes: new_unrated_population })
    }

    // Partitions the whole population into species (niches)
    fn partition<R>(self,
                    rng: &mut R,
                    threshold: f64,
                    coeff: &compatibility::CompatibilityCoefficients)
                    -> Vec<Niche>
        where R: Rng
    {
        let mut niches: Vec<Niche> = Vec::new();

        'outer: for (fitness, genome) in self.genomes.into_iter() {
            for niche in niches.iter_mut() {
                // Is this genome compatible with this niche? Test against a random genome.
                let compatible = match rng.choose(&niche.genomes) {
                    Some(&(_, ref probe)) => {
                        compatibility::compatibility(&probe.link_genes, &genome.link_genes, coeff) <
                        threshold
                    }
                    // If a niche is empty, a genome always is compatible (note that a niche can't be empyt)
                    None => true,
                };
                if compatible {
                    niche.add(fitness, genome);
                    continue 'outer;
                }
            }
            // if no compatible niche was found, insert into a new niche.
            niches.push(Niche::new_with(fitness, genome));
        }

        niches
    }
}
