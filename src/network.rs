use super::innovation::{Innovation, InnovationContainer};
use super::traits::Distance;
use super::alignment::Alignment;
use std::cmp;

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
