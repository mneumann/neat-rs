use innovation::Innovation;
use gene::Gene;
use gene_list::GeneList;
use traits::{Distance, Genotype};
use crossover::Crossover;
use rand::Rng;
use acyclic_network::{Network, NetworkMap};
pub use acyclic_network::NodeType;
use std::collections::BTreeMap;
use std::cmp;
use std::marker::PhantomData;
use mutate::MutateMethod;

#[derive(Debug, Clone)]
pub struct NodeGene<NT: NodeType> {
    pub innovation: Innovation,
    pub node_type: NT,
}

impl<NT: NodeType> Gene for NodeGene<NT> {
    fn weight_distance(&self, _other: &Self) -> f64 {
        0.0
    }

    fn innovation(&self) -> Innovation {
        self.innovation
    }
}

// To avoid node collisions we use Innovation numbers instead of node ids.
// XXX: We could omit the `innovation` and instead use (source_node_gene, target_node_gene),
// but only if duplicate links are not allowed.
#[derive(Debug, Clone)]
pub struct LinkGene {
    // The innovation number of that LinkGene
    pub innovation: Innovation,

    // This points to the NodeGene of that innovation
    pub source_node_gene: Innovation,
    // This points to the NodeGene of that innovation
    pub target_node_gene: Innovation,
    pub weight: f64,
    pub active: bool,
}

impl Gene for LinkGene {
    fn weight_distance(&self, other: &Self) -> f64 {
        self.weight - other.weight
    }

    fn innovation(&self) -> Innovation {
        self.innovation
    }
}

impl LinkGene {
    pub fn disable(&mut self) {
        assert!(self.active);
        self.active = false;
    }
}

#[derive(Clone, Debug)]
pub struct Genome<NT: NodeType> {
    link_genes: GeneList<LinkGene>,
    node_genes: GeneList<NodeGene<NT>>,
    network: NetworkMap<Innovation, NT, f64>,
}

impl<NT: NodeType> Genotype for Genome<NT> {}

impl<NT: NodeType> Genome<NT> {
    pub fn new() -> Self {
        Genome {
            link_genes: GeneList::new(),
            node_genes: GeneList::new(),
            network: NetworkMap::new(),
        }
    }

    pub fn network(&self) -> &Network<NT, f64> {
        self.network.network()
    }

    pub fn visit_node_genes<F>(&self, mut f: F)
        where F: FnMut(&NodeGene<NT>)
    {
        for node_gene_w in self.node_genes.genes().iter() {
            f(node_gene_w.as_ref())
        }
    }

    pub fn visit_active_link_genes<F>(&self, mut f: F)
        where F: FnMut(&LinkGene)
    {
        for link_gene_w in self.link_genes.genes().iter() {
            let link_gene = link_gene_w.as_ref();
            if link_gene.active {
                f(link_gene)
            }
        }
    }

    pub fn add_link(&mut self, link_gene: LinkGene) {
        self.network.add_link(link_gene.source_node_gene,
                              link_gene.target_node_gene,
                              link_gene.weight);
        self.link_genes.insert(link_gene);
    }

    pub fn add_node(&mut self, node_gene: NodeGene<NT>) {
        self.network.add_node(node_gene.innovation(), node_gene.node_type.clone());
        self.node_genes.push(node_gene);
    }

    // Uses the crossover method C to recombine left and right.
    pub fn crossover<R: Rng, C: Crossover>(left: &Self, right: &Self, c: &C, rng: &mut R) -> Self {
        let mut new_link_genes = GeneList::new();
        let mut new_node_genes = GeneList::new();

        // At first, crossover the node genes.
        c.crossover(&left.node_genes,
                    &right.node_genes,
                    &mut |gene| {
                        new_node_genes.push(gene.clone());
                    },
                    rng);

        // As the crossover of node genes might lead out some nodes which
        // we need for connections, we have to make sure that they exist.
        c.crossover(&left.link_genes,
                    &right.link_genes,
                    &mut |gene| {
                        new_link_genes.push(gene.clone());
                    },
                    rng);

        // Additionally make sure that all link genes have a correspondant node gene.
        // We do this also for disabled links!
        for link_gene in new_link_genes.genes().iter() {
            new_node_genes.insert_from_either_or_by_innovation(link_gene.as_ref().source_node_gene,
                                                               &left.node_genes,
                                                               &right.node_genes);
            new_node_genes.insert_from_either_or_by_innovation(link_gene.as_ref().target_node_gene,
                                                               &left.node_genes,
                                                               &right.node_genes);
        }

        // now we have to check that the link genes do not introduce a cycle, and remove those that do.
        let mut net = NetworkMap::new();

        // At first, add all nodes to the graph
        for node_gene in new_node_genes.genes().iter() {
            let node = node_gene.as_ref();
            net.add_node(node.innovation(), node.node_type.clone());
        }

        // now only retain those link genes, which would not introduce a cycle
        new_link_genes.retain(|link_gene| {
            if net.link_would_cycle(link_gene.source_node_gene, link_gene.target_node_gene) {
                false
            } else {
                net.add_link(link_gene.source_node_gene,
                             link_gene.target_node_gene,
                             link_gene.weight);
                true
            }
        });
        // XXX: handle disabled links

        // we might now have some additional nodes which are unconnected. leave them
        // for now.

        Genome {
            link_genes: new_link_genes,
            node_genes: new_node_genes,
            network: net,
        }
    }

    /// Returns a random link gene innovation which is active. Or None if no such exists.
    pub fn find_random_active_link_gene<R: Rng>(&self, rng: &mut R) -> Option<Innovation> {
        let mut all_active_link_genes = Vec::with_capacity(self.link_genes.len());
        for link_gene_w in self.link_genes.genes().iter() {
            let link_gene = link_gene_w.as_ref();
            if link_gene.active {
                all_active_link_genes.push(link_gene.innovation());
            }
        }
        rng.choose(&all_active_link_genes).map(|&i| i)
    }
}

pub struct GenomeDistance {
    pub excess: f64,
    pub disjoint: f64,
    pub weight: f64,
}

impl<NT: NodeType> Distance<Genome<NT>> for GenomeDistance {
    fn distance(&self, genome_left: &Genome<NT>, genome_right: &Genome<NT>) -> f64 {
        let genes_left = &genome_left.link_genes;
        let genes_right = &genome_right.link_genes;

        let max_len = cmp::max(genes_left.len(), genes_right.len());
        if max_len == 0 {
            return 0.0;
        }

        let m = genes_left.alignment_metric(&genes_right);

        self.excess * (m.excess as f64) / (max_len as f64) +
        self.disjoint * (m.disjoint as f64) / (max_len as f64) +
        self.weight *
        if m.matching > 0 {
            m.weight_distance / (m.matching as f64)
        } else {
            0.0
        }
    }
}
/// This trait is used to specialize link weight creation and node activation function creation.
pub trait ElementStrategy<NT: NodeType>
{
    fn random_link_weight<R: Rng>(rng: &mut R) -> f64;
    fn random_node_type<R: Rng>(rng: &mut R) -> NT;
}

#[derive(Debug)]
pub struct Environment<NT, S>
    where NT: NodeType,
          S: ElementStrategy<NT>
{
    node_innovation_counter: Innovation,
    link_innovation_counter: Innovation,
    // (src_node, target_node) -> link_innovation
    link_innovation_cache: BTreeMap<(Innovation, Innovation), Innovation>,
    _marker_s: PhantomData<S>,
    _marker_nt: PhantomData<NT>,
}

impl<NT, S> Environment<NT, S>
    where NT: NodeType,
          S: ElementStrategy<NT>
{
    pub fn new() -> Self {
        Environment {
            node_innovation_counter: Innovation::new(0),
            link_innovation_counter: Innovation::new(0),
            link_innovation_cache: BTreeMap::new(),
            _marker_s: PhantomData,
            _marker_nt: PhantomData,
        }
    }

    /// Adds a link to `genome` without checking if the addition of this
    /// link would introduce a cycle or would otherwise be illegal.
    fn add_link(&mut self,
                genome: &mut Genome<NT>,
                source_node: Innovation,
                target_node: Innovation,
                weight: f64) {
        let link_innovation = self.get_link_innovation(source_node, target_node);
        let link_gene = LinkGene {
            innovation: link_innovation,
            source_node_gene: source_node,
            target_node_gene: target_node,
            weight: weight,
            active: true,
        };
        genome.add_link(link_gene)
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

    pub fn mutate<R: Rng>(&mut self,
                          genome: &Genome<NT>,
                          method: MutateMethod,
                          rng: &mut R)
                          -> Option<Genome<NT>> {
        match method {
            MutateMethod::ModifyWeight => {
                // XXX
                None
            }
            MutateMethod::AddConnection => self.mutate_add_connection(genome, rng),
            MutateMethod::AddNode => self.mutate_add_node(genome, rng),
        }
    }

    pub fn mutate_add_connection<R: Rng>(&mut self,
                                         genome: &Genome<NT>,
                                         rng: &mut R)
                                         -> Option<Genome<NT>> {
        genome.network.find_random_unconnected_link_no_cycle(rng).map(|(&src, &target)| {
            let mut offspring = genome.clone();
            // Add new link to the offspring genome
            self.add_link(&mut offspring, src, target, S::random_link_weight(rng));
            offspring
        })
    }

    pub fn add_node_to_genome(&mut self, genome: &mut Genome<NT>, node_type: NT) {
        genome.add_node(NodeGene {
            innovation: self.node_innovation_counter.next().unwrap(),
            node_type: node_type,
        });
    }

    /// choose a random link. split it in half.
    /// XXX: activate if link is inactive?
    pub fn mutate_add_node<R: Rng>(&mut self,
                                   genome: &Genome<NT>,
                                   rng: &mut R)
                                   -> Option<Genome<NT>> {
        genome.find_random_active_link_gene(rng).map(|link_innov| {
            // split link in half.
            let mut offspring = genome.clone();
            let new_node_innovation = self.new_node_innovation();
            // add new node
            offspring.add_node(NodeGene {
                innovation: new_node_innovation,
                node_type: S::random_node_type(rng),
            });

            // disable `link_innov` in offspring
            // we keep this gene (but disable it), because this allows us to have a structurally
            // compatible genome to the old one, as disabled genes are taken into account for
            // the genomic distance measure.
            let (orig_src_node, orig_target_node) = {
                let mut orig_link = offspring.link_genes
                                             .find_by_innovation_mut(link_innov)
                                             .unwrap();
                orig_link.disable();
                // add two new link innovations with the new node in the middle.
                // XXX: Choose random weights? Or split weight? We use random weights for now.
                (orig_link.source_node_gene, orig_link.target_node_gene)
            };
            // adding these two links cannot create a cycle.
            self.add_link(&mut offspring,
                          orig_src_node,
                          new_node_innovation,
                          S::random_link_weight(rng));
            self.add_link(&mut offspring,
                          new_node_innovation,
                          orig_target_node,
                          S::random_link_weight(rng));
            offspring
        })
    }
}
