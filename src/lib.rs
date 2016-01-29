use std::collections::BTreeMap;
use std::cmp;
use innovation::Innovation;

mod innovation;

trait Gene {
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

// To avoid node collisions we use Innovation numbers instead. Every newly created node gets it's
// own innovation number.
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

// always take excess/disjoint genes if they appear on the fitter parent.
// if they appear on the less fit parent, take them only to a specified
// probability

struct Genome {
    link_genes: BTreeMap<Innovation, LinkGene>,
    node_genes: BTreeMap<Innovation, NodeGene>,
}

struct Coefficients {
    excess: f64,
    disjoint: f64,
    weight: f64,
}

fn genes_compatibility<T: Gene>(genes1: &BTreeMap<Innovation, T>,
                                genes2: &BTreeMap<Innovation, T>,
                                c: &Coefficients)
                                -> f64 {
    let max_len = cmp::max(genes1.len(), genes2.len());
    assert!(max_len > 0);

    let range1 = innovation::innovation_range(genes1);
    let range2 = innovation::innovation_range(genes2);

    // calculate the number of excess genes of
    let mut excess = 0;
    let mut disjoint = 0;
    let mut matching = 0;
    let mut weight_dist = 0.0;

    for (innov1, gene) in genes1.iter() {
        if innov1.is_within(&range2) {
            match genes2.get(innov1) {
                Some(other_gene) => {
                    matching += 1;
                    weight_dist += gene.weight_distance(other_gene);
                }
                None => {
                    disjoint += 1;
                }
            }
        } else {
            excess += 1;
        }
    }

    for innov2 in genes2.keys() {
        if innov2.is_within(&range1) {
            if !genes1.contains_key(innov2) {
                disjoint += 1;
            }
        } else {
            excess += 1;
        }
    }

    assert!(2 * matching + disjoint + excess == genes1.len() + genes2.len());

    c.excess * (excess as f64) / (max_len as f64) +
    c.disjoint * (disjoint as f64) / (max_len as f64) +
    if matching > 0 {
        weight_dist / (matching as f64)
    } else {
        0.0
    }
}
