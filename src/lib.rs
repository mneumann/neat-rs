extern crate rand;

use std::collections::BTreeMap;
use std::cmp;
use innovation::Innovation;
use rand::{Rng, Closed01};

mod innovation;

trait Gene: Clone {
    fn weight_distance(&self, other: &Self) -> f64;
}

enum NodeType {
    Input,
    Output,
    Hidden,
}

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

struct Genome {
    link_genes: BTreeMap<Innovation, LinkGene>,
    node_genes: BTreeMap<Innovation, NodeGene>,
}

enum Correlation<'a, T: 'a> {
    Matching(&'a T, &'a T),
    DisjointLeft(&'a T),
    DisjointRight(&'a T),
    ExcessLeft(&'a T),
    ExcessRight(&'a T),
}

/// Correlate the genes in ```genes_left``` with those in ```genes_right```.
fn correlate<'a, T, F>(genes_left: &'a BTreeMap<Innovation, T>,
                       genes_right: &'a BTreeMap<Innovation, T>,
                       f: &mut F)
    where F: FnMut(Innovation, Correlation<'a, T>)
{
    let range_left = innovation::innovation_range(genes_left);
    let range_right = innovation::innovation_range(genes_right);

    for (innov_left, gene_left) in genes_left.iter() {
        if innov_left.is_within(&range_right) {
            match genes_right.get(innov_left) {
                Some(gene_right) => f(*innov_left, Correlation::Matching(gene_left, gene_right)),
                None => f(*innov_left, Correlation::DisjointLeft(gene_left)),
            }
        } else {
            f(*innov_left, Correlation::ExcessLeft(gene_left))
        }
    }

    for (innov_right, gene_right) in genes_right.iter() {
        if innov_right.is_within(&range_left) {
            if !genes_left.contains_key(innov_right) {
                f(*innov_right, Correlation::DisjointRight(gene_right))
            }
        } else {
            f(*innov_right, Correlation::ExcessRight(gene_right))
        }
    }
}

struct CompatibilityCoefficients {
    excess: f64,
    disjoint: f64,
    weight: f64,
}

/// Calculates the compatibility between two gene lists.
fn compatibility<T: Gene>(genes1: &BTreeMap<Innovation, T>,
                              genes2: &BTreeMap<Innovation, T>,
                              coeff: &CompatibilityCoefficients)
                              -> f64 {
    let max_len = cmp::max(genes1.len(), genes2.len());
    assert!(max_len > 0);

    let mut matching = 0;
    let mut disjoint = 0;
    let mut excess = 0;
    let mut weight_dist = 0.0;

    correlate(genes1,
              genes2,
              &mut |_, correlation| {
                  match correlation {
                      Correlation::Matching(gene_left, gene_right) => {
                          matching += 1;
                          weight_dist += gene_left.weight_distance(gene_right).abs();
                      }
                      Correlation::DisjointLeft(_) | Correlation::DisjointRight(_) => {
                          disjoint += 1;
                      }
                      Correlation::ExcessLeft(_) | Correlation::ExcessRight(_) => {
                          excess += 1;
                      }
                  }
              });

    assert!(2 * matching + disjoint + excess == genes1.len() + genes2.len());

    coeff.excess * (excess as f64) / (max_len as f64) +
    coeff.disjoint * (disjoint as f64) / (max_len as f64) +
    coeff.weight *
    if matching > 0 {
        weight_dist / (matching as f64)
    } else {
        0.0
    }
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
        v < prob.0
    } else {
        true
    }
}

/// We assume ```parent1``` to be the fitter parent. Takes gene 
/// either from ```parent1``` or ```parent2``` according to
/// the probabilities specified and the relative fitness of the parents.
fn crossover<T: Gene, R: Rng>(parent1: &BTreeMap<Innovation, T>,
                              parent2: &BTreeMap<Innovation, T>,
                              p: &CrossoverProbabilities,
                              rng: &mut R)
                              -> BTreeMap<Innovation, T> {

    let mut offspring = BTreeMap::new();

    correlate(parent1,
              parent2,
              &mut |innov, correlation| {
                  match correlation {
                      Correlation::Matching(gene_left, gene_right) => {
                          if is_probable(&p.prob_match1, rng) {
                              offspring.insert(innov, gene_left.clone());
                          } else {
                              offspring.insert(innov, gene_right.clone());
                          }
                      }

                      Correlation::DisjointLeft(gene_left) => {
                          if is_probable(&p.prob_disjoint1, rng) {
                              offspring.insert(innov, gene_left.clone());
                          }
                      }

                      Correlation::DisjointRight(gene_right) => {
                          if is_probable(&p.prob_disjoint2, rng) {
                              offspring.insert(innov, gene_right.clone());
                          }
                      }

                      Correlation::ExcessLeft(gene_left) => {
                          if is_probable(&p.prob_excess1, rng) {
                              offspring.insert(innov, gene_left.clone());
                          }
                      }

                      Correlation::ExcessRight(gene_right) => {
                          if is_probable(&p.prob_excess2, rng) {
                              offspring.insert(innov, gene_right.clone());
                          }
                      }
                  }
              });

    offspring
}
