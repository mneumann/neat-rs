use rand::{Rng, Closed01};
use super::alignment::Alignment;
use super::innovation::InnovationContainer;
use super::is_probable;

// XXX: Move to traits. Abstract InnovationContainer away.
pub trait Crossover {
    /// Performs a crossover between `parent_left` and `parent_right`
    /// resulting in a single offspring. It can be assumed that
    /// `parent_left` is the fitter parent.
    fn crossover<T: Clone, R: Rng>(&self,
                                   parent_left: &InnovationContainer<T>,
                                   parent_right: &InnovationContainer<T>,
                                   rng: &mut R)
                                   -> InnovationContainer<T>;
}

/// A specific form of crossover where the probabilities below determine from
/// parent a gene is taken.
///
/// XXX: It's probably faster to not use floats here.
pub struct ProbabilisticCrossover {
    /// Probability to take a matching gene from the fitter (left) parent.
    pub prob_match_left: Closed01<f32>,

    /// Probability to take a disjoint gene from the fitter (left) parent.
    pub prob_disjoint_left: Closed01<f32>,

    /// Probability to take an excess gene from the fitter (left) parent.
    pub prob_excess_left: Closed01<f32>,

    /// Probability to take a disjoint gene from the less fit (right) parent.
    pub prob_disjoint_right: Closed01<f32>,

    /// Probability to take an excess gene from the less fit (right) parent.
    pub prob_excess_right: Closed01<f32>,
}

impl Crossover for ProbabilisticCrossover {
    /// `parent_left` is the fitter parent. Take gene
    /// either from `parent_left` or `parent_right` according to
    /// the specified probabilities and the relative fitness of the parents.
    fn crossover<T: Clone, R: Rng>(&self,
                                   parent_left: &InnovationContainer<T>,
                                   parent_right: &InnovationContainer<T>,
                                   rng: &mut R)
                                   -> InnovationContainer<T> {

        let mut offspring = InnovationContainer::new();

        parent_left.align(parent_right,
                          &mut |innov, alignment| {
                              match alignment {
                                  Alignment::Match(gene_left, gene_right) => {
                                      if is_probable(&self.prob_match_left, rng) {
                                          offspring.insert(innov, gene_left.clone());
                                      } else {
                                          offspring.insert(innov, gene_right.clone());
                                      }
                                  }

                                  Alignment::DisjointLeft(gene_left) => {
                                      if is_probable(&self.prob_disjoint_left, rng) {
                                          offspring.insert(innov, gene_left.clone());
                                      }
                                  }

                                  Alignment::DisjointRight(gene_right) => {
                                      if is_probable(&self.prob_disjoint_right, rng) {
                                          offspring.insert(innov, gene_right.clone());
                                      }
                                  }

                                  Alignment::ExcessLeft(gene_left) => {
                                      if is_probable(&self.prob_excess_left, rng) {
                                          offspring.insert(innov, gene_left.clone());
                                      }
                                  }

                                  Alignment::ExcessRight(gene_right) => {
                                      if is_probable(&self.prob_excess_right, rng) {
                                          offspring.insert(innov, gene_right.clone());
                                      }
                                  }
                              }
                          });

        offspring
    }
}
