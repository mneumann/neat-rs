use super::innovation::{Innovation, InnovationContainer};
use super::traits::{Distance, Genotype};
use super::alignment::Alignment;
use std::cmp;

#[derive(Debug, Copy, Clone)]
pub enum NodeType {
    Input,
    Output,
    Hidden,
}

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub node_type: NodeType,
}

// To avoid node collisions we use Innovation numbers instead of node
// ids.
#[derive(Debug, Clone)]
pub struct LinkGene {
    // This points to the NodeGene of that innovation
    pub source_node_gene: Innovation,
    // This points to the NodeGene of that innovation
    pub target_node_gene: Innovation,
    pub weight: f64,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct NetworkGenome {
    pub link_genes: InnovationContainer<LinkGene>,
    pub node_genes: InnovationContainer<NodeGene>,
}

impl Genotype for NetworkGenome {}

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

pub struct NetworkGenomeDistance {
    pub l: LinkGeneListDistance,
}

impl Distance<NetworkGenome> for NetworkGenomeDistance {
    fn distance(&self, genome_left: &NetworkGenome, genome_right: &NetworkGenome) -> f64 {
        self.l.distance(&genome_left.link_genes, &genome_right.link_genes)
    }
}
