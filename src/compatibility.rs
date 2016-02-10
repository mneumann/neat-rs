use super::Gene;
use super::innovation::InnovationContainer;
use super::alignment::Alignment;
use std::cmp;

/// The specific weighting of the `compatibility` function.
pub struct CompatibilityCoefficients {
    pub excess: f64,
    pub disjoint: f64,
    pub weight: f64,
}

/// Calculates the compatibility between two gene lists.
pub fn compatibility<T: Gene>(genes_left: &InnovationContainer<T>,
                              genes_right: &InnovationContainer<T>,
                              coeff: &CompatibilityCoefficients)
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
                                 weight_dist += gene_left.weight_distance(gene_right)
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

    coeff.excess * (excess as f64) / (max_len as f64) +
    coeff.disjoint * (disjoint as f64) / (max_len as f64) +
    coeff.weight *
    if matching > 0 {
        weight_dist / (matching as f64)
    } else {
        0.0
    }
}
