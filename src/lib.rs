#![feature(iter_arith, zero_one)]

extern crate rand;

use std::cmp;
use innovation::{Innovation, InnovationContainer};
use alignment::Alignment;
use rand::{Rng, Closed01};
use mate::Mate;
use traits::Distance;
use traits::Genotype;
use fitness::Fitness;

mod traits;
mod innovation;
mod selection;
mod alignment;
mod crossover;
mod mate;
mod fitness;

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

#[derive(Clone)]
struct Genome {
    link_genes: InnovationContainer<LinkGene>,
    node_genes: InnovationContainer<NodeGene>,
}

struct LinkGeneWeightDistance;

impl Distance<LinkGene> for LinkGeneWeightDistance {
    fn distance(&self, a: &LinkGene, b: &LinkGene) -> f64 {
        a.weight - b.weight
    }
}

pub struct LinkGeneListDistance {
    pub excess: f64,
    pub disjoint: f64,
    pub weight: f64,
}

impl Distance<InnovationContainer<LinkGene>> for LinkGeneListDistance {
    fn distance(&self,
                genes_left: &InnovationContainer<LinkGene>,
                genes_right: &InnovationContainer<LinkGene>)
                -> f64 {
        let max_len = cmp::max(genes_left.len(), genes_right.len());
        assert!(max_len > 0);

        let mut matching = 0;
        let mut disjoint = 0;
        let mut excess = 0;
        let mut weight_dist = 0.0;

        genes_left.align(genes_right,
                         &mut |_, alignment| {
                             match alignment {
                                 Alignment::Match(gene_left, gene_right) => {
                                     matching += 1;
                                     weight_dist += LinkGeneWeightDistance.distance(gene_left,
                                                                                    gene_right)
                                                                          .abs();
                                 }
                                 Alignment::DisjointLeft(_) | Alignment::DisjointRight(_) => {
                                     disjoint += 1;
                                 }
                                 Alignment::ExcessLeft(_) | Alignment::ExcessRight(_) => {
                                     excess += 1;
                                 }
                             }
                         });

        assert!(2 * matching + disjoint + excess == genes_left.len() + genes_right.len());

        self.excess * (excess as f64) / (max_len as f64) +
        self.disjoint * (disjoint as f64) / (max_len as f64) +
        self.weight *
        if matching > 0 {
            weight_dist / (matching as f64)
        } else {
            0.0
        }
    }
}

pub struct GenomeDistance {
    pub l: LinkGeneListDistance,
}

impl Distance<Genome> for GenomeDistance {
    fn distance(&self, genome_left: &Genome, genome_right: &Genome) -> f64 {
        self.l.distance(&genome_left.link_genes, &genome_right.link_genes)
    }
}

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
        Niche {
            genomes: vec![(fitness, genome)],
            fitness_sum: fitness,
        }
    }

    fn mean_fitness(&self) -> Fitness {
        self.fitness_sum / Fitness::new(self.len() as f64)
    }

    fn add(&mut self, fitness: Fitness, genome: Box<Genome>) {
        self.genomes.push((fitness, genome));
        self.fitness_sum = self.fitness_sum + fitness;
    }

    fn sort(mut self) -> Self {
        (&mut self.genomes).sort_by(|a, b| a.0.cmp(&b.0));
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
                                  -> (RatedPopulation, UnratedPopulation)
        where R: Rng,
              C: Distance<Genome>,
              M: Mate
    {
        assert!(elite_percentage.0 <= selection_percentage.0); // XXX
        assert!(self.len() > 0);
        let niches = self.partition(rng, threshold, compatibility);
        assert!(niches.len() > 0);
        let total_mean: Fitness = niches.iter().map(|n| n.mean_fitness()).sum();
        // XXX: total_mean = 0.0?

        let mut new_unrated_population = Vec::new();
        let mut new_rated_population = Vec::new();

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

                    let offspring = mate.mate(&sorted_niche.genomes[parent1].1,
                                              &sorted_niche.genomes[parent2].1,
                                              rng);
                    // let offspring = sorted_niche.genomes[parent1].1.clone();
                    new_unrated_population.push(Box::new(offspring));
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
    fn partition<R, C>(self, rng: &mut R, threshold: f64, compatibility: &C) -> Vec<Niche>
        where R: Rng,
              C: Distance<Genome>
    {
        let mut niches: Vec<Niche> = Vec::new();

        'outer: for (fitness, genome) in self.genomes.into_iter() {
            for niche in niches.iter_mut() {
                // Is this genome compatible with this niche? Test against a random genome.
                let compatible = match rng.choose(&niche.genomes) {
                    Some(&(_, ref probe)) => compatibility.distance(probe, &genome) < threshold,
                    // If a niche is empty, a genome always is compatible (note that a niche can't be empyt)
                    None => true,
                };
                if compatible {
                    niche.add(fitness, genome);
                    continue 'outer;
                }
            }
            // if no compatible niche was found, create a new niche containing this genome.
            niches.push(Niche::new_with(fitness, genome));
        }

        niches
    }
}
