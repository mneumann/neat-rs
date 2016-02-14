use super::innovation::{Innovation, InnovationContainer};
use super::traits::{Distance, Genotype};
use super::alignment::Alignment;
use std::cmp;
use super::crossover::Crossover;
use rand::Rng;
use fixedbitset::FixedBitSet;
use std::collections::BTreeMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Output,
    Hidden,
}

impl Default for NodeType {
    fn default() -> Self {
        NodeType::Hidden
    }
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

impl LinkGene {
    pub fn disable(&mut self) {
        assert!(self.active);
        self.active = false;
    }
}

#[derive(Debug, Clone)]
pub struct NetworkGenome {
    pub link_genes: InnovationContainer<LinkGene>,
    pub node_genes: InnovationContainer<NodeGene>,
}

#[derive(Debug, Clone)]
struct AdjMatrix {
    n: usize,
    m: FixedBitSet,
}

impl AdjMatrix {
    fn new(n: usize) -> AdjMatrix {
        AdjMatrix {
            n: n,
            m: FixedBitSet::with_capacity(n * n),
        }
    }

    fn set(&mut self, i: usize, j: usize) {
        self.m.insert(i * self.n + j);
    }

    fn contains(&self, i: usize, j: usize) -> bool {
        self.m.contains(i * self.n + j)
    }

    /// This is O(n**4) in worst case.
    /// XXX: This needs a test for correctness
    fn transitive_closure(mut self) -> AdjMatrix {
        loop {
            let mut counts = 0;
            for i in 0..self.n {
                for j in 0..self.n {
                    if self.contains(i, j) {
                        // look column for j
                        for k in 0..self.n {
                            if self.contains(j, k) {
                                if !self.contains(i, k) {
                                    self.set(i, k);
                                    counts += 1;
                                }
                            }
                        }
                    }
                }
            }
            if counts == 0 {
                break;
            }
        }
        self
    }

    fn unconnected_pairs_no_cycle(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.n {
            for j in 0..self.n {
                if i != j {
                    if !(self.contains(i, j) || self.contains(j, i)) {
                        // make sure we don't introduce a cycle
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs
    }
}

impl NetworkGenome {
    // Uses the crossover method C to recombine left and right.
    pub fn crossover<R: Rng, C: Crossover>(left: &Self, right: &Self, c: &C, rng: &mut R) -> Self {
        let new_link_genes = c.crossover(&left.link_genes, &right.link_genes, rng);
        let mut new_node_genes = c.crossover(&left.node_genes, &right.node_genes, rng);

        // Additionally make sure that all link genes have a correspondant node gene.
        // We do this also for disabled links!
        for link in new_link_genes.map.values() {
            new_node_genes.insert_from_either_or(link.source_node_gene,
                                                 &left.node_genes,
                                                 &right.node_genes);
            new_node_genes.insert_from_either_or(link.target_node_gene,
                                                 &left.node_genes,
                                                 &right.node_genes);
        }

        NetworkGenome {
            link_genes: new_link_genes,
            node_genes: new_node_genes,
        }
    }

    /// Constructs the adjacency matrix from the genome.
    fn adjacency_matrix(&self) -> (AdjMatrix, Vec<Innovation>) {
        // map the node innovation to the index in the adjacency matrix
        let mut map_to_indices: BTreeMap<Innovation, usize> = BTreeMap::new();
        // and the reverse map from index to Innovation
        let mut rev_map: Vec<Innovation> = Vec::new();

        for &node_innov in self.node_genes.map.keys() {
            let next_idx = rev_map.len();
            rev_map.push(node_innov);
            let prev = map_to_indices.insert(node_innov, next_idx);
            assert!(prev.is_none());
        }

        assert!(rev_map.len() == map_to_indices.len());

        let n = map_to_indices.len();
        let mut adj_matrix = AdjMatrix::new(n);
        for link in self.link_genes.map.values() {
            if link.active {
                adj_matrix.set(map_to_indices[&link.source_node_gene],
                               map_to_indices[&link.target_node_gene]);
            }
        }

        (adj_matrix, rev_map)
    }

    /// Returns a random link gene innovation which is active. Or None if no such exists.
    pub fn find_random_active_link_gene<R: Rng>(&self, rng: &mut R) -> Option<Innovation> {
        let mut all_active_link_innovations = Vec::with_capacity(self.link_genes.len());
        for (&innov, link) in self.link_genes.map.iter() {
            if link.active {
                all_active_link_innovations.push(innov);
            }
        }
        rng.choose(&all_active_link_innovations).map(|&i| i)
    }

    /// Returns two node innvovations which are not yet connected and which does not
    /// create a cycle. Return `None` if no such connection exists.
    pub fn find_random_unconnected_pair<R: Rng>(&self,
                                                rng: &mut R)
                                                -> Option<(Innovation, Innovation)> {
        // Construct binary adjacency matrix
        let (adj_matrix, rev_map) = self.adjacency_matrix();

        // Construct an array of all currently unconnected nodes.
        let unconnected = adj_matrix.transitive_closure().unconnected_pairs_no_cycle();

        // Out of all unconnected pairs, choose a random one.
        rng.choose(&unconnected).map(|&(source, target)| {
            // reverse map src and target to the node innovation
            (rev_map[source], rev_map[target])
        })
    }
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
        if max_len == 0 {
            return 0.0;
        }

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

pub struct Environment {
    node_innovation_counter: Innovation,
    link_innovation_counter: Innovation,
    // (src_node, target_node) -> link_innovation
    link_innovation_cache: BTreeMap<(Innovation, Innovation), Innovation>,
}

impl Environment {
    pub fn new() -> Environment {
        Environment {
            node_innovation_counter: Innovation::new(0),
            link_innovation_counter: Innovation::new(0),
            link_innovation_cache: BTreeMap::new(),
        }
    }

    // XXX
    fn random_link_weight<R: Rng>(&mut self, rng: &mut R) -> f64 {
        // XXX Choose a weight between -1 and 1?
        rng.gen()
    }

    fn add_link<R: Rng>(&mut self,
                        genome: &mut NetworkGenome,
                        source_node: Innovation,
                        target_node: Innovation,
                        rng: &mut R) {
        let link_gene = LinkGene {
            source_node_gene: source_node,
            target_node_gene: target_node,
            weight: self.random_link_weight(rng),
            active: true,
        };
        self.add_link_gene(genome, link_gene);
    }

    fn add_link_gene(&mut self, genome: &mut NetworkGenome, link_gene: LinkGene) {
        let link_innovation = self.get_link_innovation(link_gene.source_node_gene,
                                                       link_gene.target_node_gene);
        genome.link_genes.insert_or_replace(link_innovation, link_gene);
    }

    fn get_link_innovation(&mut self,
                           source_node_gene: Innovation,
                           target_node_gene: Innovation)
                           -> Innovation {
        let key = (source_node_gene, target_node_gene);
        if let Some(&cached_innovation) = self.link_innovation_cache.get(&key) {
            return cached_innovation;
        }
        let new_innovation = self.link_innovation_counter.next().unwrap();
        self.link_innovation_cache.insert(key, new_innovation);
        new_innovation
    }

    fn new_node_innovation(&mut self) -> Innovation {
        self.node_innovation_counter.next().unwrap()
    }

    pub fn mutate_add_connection<R: Rng>(&mut self,
                                         genome: &NetworkGenome,
                                         rng: &mut R)
                                         -> Option<NetworkGenome> {
        genome.find_random_unconnected_pair(rng).map(|(src, target)| {
            let mut offspring = genome.clone();
            // Add new link to the offspring genome
            self.add_link(&mut offspring, src, target, rng);
            offspring
        })
    }

    /// choose a random link. split it in half.
    pub fn mutate_add_node<R: Rng>(&mut self,
                                   genome: &NetworkGenome,
                                   rng: &mut R)
                                   -> Option<NetworkGenome> {
        genome.find_random_active_link_gene(rng).map(|link_innov| {
            // split link in half.
            let mut offspring = genome.clone();
            let new_node_innovation = self.new_node_innovation();
            // add new node
            offspring.node_genes.insert(new_node_innovation,
                                        NodeGene { node_type: NodeType::Hidden });
            // disable `link_innov` in offspring
            // we keep this gene (but disable it), because this allows us to have a structurally
            // compatible genome to the old one, as disabled genes are taken into account for
            // the genomic distance measure.
            offspring.link_genes.get_mut(&link_innov).unwrap().disable();
            // add two new link innovations with the new node in the middle.
            // XXX: Choose random weights? Or split weight? We use random weights for now.
            let (orig_src_node, orig_target_node) = {
                let orig_link = offspring.link_genes.get(&link_innov).unwrap();
                (orig_link.source_node_gene, orig_link.target_node_gene)
            };
            self.add_link(&mut offspring, orig_src_node, new_node_innovation, rng);
            self.add_link(&mut offspring, new_node_innovation, orig_target_node, rng);
            offspring
        })
    }

    // Generates a NetworkGenome with `n_inputs` input nodes and `n_outputs` output nodes.
    // The genome will not have any link nodes.
    pub fn generate_genome(&mut self, n_inputs: usize, n_outputs: usize) -> NetworkGenome {
        assert!(n_inputs > 0 && n_outputs > 0);
        let mut nodes = InnovationContainer::new();
        for _ in 0..n_inputs {
            nodes.insert(self.node_innovation_counter.next().unwrap(),
                         NodeGene { node_type: NodeType::Input });
        }
        assert!(nodes.len() == n_inputs);
        for _ in 0..n_outputs {
            nodes.insert(self.node_innovation_counter.next().unwrap(),
                         NodeGene { node_type: NodeType::Output });
        }
        assert!(nodes.len() == n_inputs + n_outputs);
        NetworkGenome {
            link_genes: InnovationContainer::new(),
            node_genes: nodes,
        }
    }
}
