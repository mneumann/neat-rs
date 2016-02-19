use rand::{Rng, Closed01};
use prob::is_probable;
use gene::Gene;
use gene_list::GeneList;
use alignment::{Alignment, align};

pub trait Crossover {
    fn crossover<T: Gene, R: Rng, F: FnMut(&T)>(&self,
                                                parent_left: &GeneList<T>,
                                                parent_right: &GeneList<T>,
                                                construct: &mut F,
                                                rng: &mut R);
}

/// A specific form of crossover where the probabilities below determine from
/// which parent a gene is taken.
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
    ///
    /// The individual genes are passed to `construct`, which can further decide
    /// if that gene is taken into the offspring or not.
    fn crossover<T: Gene, R: Rng, F: FnMut(&T)>(&self,
                                                parent_left: &GeneList<T>,
                                                parent_right: &GeneList<T>,
                                                construct: &mut F,
                                                rng: &mut R) {
        align(parent_left.genes(),
              parent_right.genes(),
              &mut |alignment| {
                  match alignment {
                      Alignment::Match(gene_left, gene_right) => {
                          if is_probable(&self.prob_match_left, rng) {
                              construct(gene_left.as_ref());
                          } else {
                              construct(gene_right.as_ref());
                          }
                      }

                      Alignment::DisjointLeft(gene_left) => {
                          if is_probable(&self.prob_disjoint_left, rng) {
                              construct(gene_left.as_ref());
                          }
                      }

                      Alignment::DisjointRight(gene_right) => {
                          if is_probable(&self.prob_disjoint_right, rng) {
                              construct(gene_right.as_ref());
                          }
                      }

                      Alignment::ExcessLeftHead(gene_left) |
                      Alignment::ExcessLeftTail(gene_left) => {
                          if is_probable(&self.prob_excess_left, rng) {
                              construct(gene_left.as_ref());
                          }
                      }

                      Alignment::ExcessRightHead(gene_right) |
                      Alignment::ExcessRightTail(gene_right) => {
                          if is_probable(&self.prob_excess_right, rng) {
                              construct(gene_right.as_ref());
                          }
                      }
                  }
              });
    }
}
