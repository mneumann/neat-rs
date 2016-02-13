extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml as gml;

use neat::population::{Population, Rated, Unrated, Individual, Runner};
use neat::network::{NetworkGenome, NetworkGenomeDistance, LinkGeneListDistance, LinkGene, NodeGene, NodeType};
use neat::innovation::{Innovation, InnovationContainer};
use neat::fitness::Fitness;
use neat::traits::Mate;
use neat::crossover::ProbabilisticCrossover;
use neat::prob::is_probable;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, IgnoreNodeColors};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use std::collections::BTreeMap;
use rand::{Rng, Closed01};
use rand::distributions::{WeightedChoice, Weighted, IndependentSample};
use std::marker::PhantomData;

fn load_graph(graph_file: &str) -> OwnedGraph {
    use std::fs::File;
    use std::io::Read;

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = gml::parse_gml(&graph_s,
                               &|_| -> Option<()> { Some(()) },
                               &|_| -> Option<()> { Some(()) })
                    .unwrap();
    OwnedGraph::from_petgraph(&graph)
}

fn genome_to_graph(genome: &NetworkGenome) -> OwnedGraph {
    let mut builder = GraphBuilder::new();

    for (&innov, node) in genome.node_genes.map.iter() {
        // make sure the node exists, even if there are no connection to it.
        let _ = builder.add_or_replace_node(innov.get());
    }

    for link in genome.link_genes.map.values() {
        if link.active {
            builder.add_edge_unweighted(link.source_node_gene.get(), link.target_node_gene.get());
        }
    }

    return builder.graph();
}

#[derive(Debug)]
struct FitnessEvaluator {
    target_graph: OwnedGraph,
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &NetworkGenome) -> f32 {
        let graph = genome_to_graph(genome);
        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, IgnoreNodeColors);
        s.iterate(50, 0.01);
        s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get()
    }
}

struct Environment {
    node_innovation_counter: Innovation,
    link_innovation_counter: Innovation,
    // (src_node, target_node) -> link_innovation
    link_innovation_cache: BTreeMap<(Innovation, Innovation), Innovation>,
}

impl Environment {
    fn new() -> Environment {
        Environment {
            node_innovation_counter: Innovation::new(0),
            link_innovation_counter: Innovation::new(0),
            link_innovation_cache: BTreeMap::new(),
        }
    }

    fn random_link_weight<R: Rng>(&mut self, rng: &mut R) -> f64 {
        // XXX Choose a weight between -1 and 1?
        rng.gen()
    }

    fn insert_new_link<R: Rng>(&mut self, genome: &mut NetworkGenome, source_node: Innovation, target_node: Innovation, rng: &mut R) {
        let link_gene = LinkGene {
            source_node_gene: source_node,
            target_node_gene: target_node,
            weight: self.random_link_weight(rng),
            active: true,
        };
        self.insert_new_link_gene(genome, link_gene);
    }

    fn insert_new_link_gene(&mut self, genome: &mut NetworkGenome, link_gene: LinkGene) {
        let new_link_innovation = self.new_link_innovation(link_gene.source_node_gene, link_gene.target_node_gene);
        genome.link_genes.insert(new_link_innovation, link_gene);
    }

    fn mutate_add_connection<R: Rng>(&mut self, genome: &NetworkGenome, rng: &mut R) -> Option<NetworkGenome> { 
        genome.find_unconnected_pair(rng).map(|(src, target)| {
            let mut offspring = genome.clone();
            // Add new link to the offspring genome
            self.insert_new_link(&mut offspring, src, target, rng);
            offspring
        })
    }

    /// choose a random link. split it in half.
    fn mutate_add_node<R: Rng>(&mut self, genome: &NetworkGenome, rng: &mut R) -> Option<NetworkGenome> {
        if let Some(link_innov) = genome.find_random_active_link_gene(rng) {
            // split link in half.
            let mut offspring = genome.clone();
            let new_node_innovation = self.new_node_innovation();
            // add new node
            offspring.node_genes.insert(new_node_innovation, NodeGene{node_type: NodeType::Hidden});
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
            self.insert_new_link(&mut offspring, orig_src_node, new_node_innovation, rng);
            self.insert_new_link(&mut offspring, new_node_innovation, orig_target_node, rng);
            Some(offspring)
        } else {
            None
        }
    }

    fn new_node_innovation(&mut self) -> Innovation {
        self.node_innovation_counter.next().unwrap()
    }

    fn new_link_innovation(&mut self, source_node_gene: Innovation, target_node_gene: Innovation) -> Innovation {
        let key = (source_node_gene, target_node_gene);
        if let Some(&cached_innovation) = self.link_innovation_cache.get(&key) {
            return cached_innovation;
        }
        let new_innovation = self.link_innovation_counter.next().unwrap();
        self.link_innovation_cache.insert(key, new_innovation);
        new_innovation
    }

    // Generates a NetworkGenome with `n_inputs` input nodes and `n_outputs` output nodes.
    // The genome will not have any link nodes.
    fn generate_genome(&mut self, n_inputs: usize, n_outputs: usize) -> NetworkGenome {
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

const POP_SIZE: usize = 100;
const INPUTS: usize = 2;
const OUTPUTS: usize = 3;

#[derive(Debug, Copy, Clone)]
enum RecombinationMethod {
    MutateWeight,
    MutateStructure,
    Crossover,
}

struct Mater<'a> {
    w_mutate_weight: u32,
    w_mutate_structure: u32,
    w_crossover: u32,
    env: &'a mut Environment,
}

impl<'a> Mate<NetworkGenome> for Mater<'a> {
    // Add an argument that descibes whether both genomes are of equal fitness.
    // Pass individual, which includes the fitness.
    fn mate<R: Rng>(&mut self,
                    parent_left: &NetworkGenome,
                    parent_right: &NetworkGenome,
                    rng: &mut R)
                    -> NetworkGenome {
        assert!(self.w_mutate_weight + self.w_mutate_structure + self.w_crossover  > 0);
        let mut items = [
            Weighted{weight: self.w_mutate_weight, item: RecombinationMethod::MutateWeight},
            Weighted{weight: self.w_mutate_structure, item: RecombinationMethod::MutateStructure},
            Weighted{weight: self.w_crossover, item: RecombinationMethod::Crossover},
        ];
        let wc = WeightedChoice::new(&mut items);

        match wc.ind_sample(rng) {
            RecombinationMethod::MutateWeight => {
                // TODO
               parent_left.clone()
            }
            RecombinationMethod::MutateStructure => {
                if is_probable(&Closed01(0.5), rng) {
                    // AddConnection
                    let offspring = self.env.mutate_add_connection(parent_left, rng).
                        or_else(|| self.env.mutate_add_connection(parent_right, rng)).
                        unwrap_or_else(|| parent_left.clone());
                    offspring
                } else {
                    // AddNode (split existing connection)
                    let offspring = self.env.mutate_add_node(parent_left, rng).
                        or_else(|| self.env.mutate_add_node(parent_right, rng)).
                        unwrap_or_else(|| parent_left.clone());
                    offspring
                }
            }
            RecombinationMethod::Crossover => {
                let x = ProbabilisticCrossover {
                    prob_match_left: Closed01(0.5), // NEAT always selects a random parent for matching genes
                    prob_disjoint_left: Closed01(0.9),
                    prob_excess_left: Closed01(0.9),
                    prob_disjoint_right: Closed01(0.15),
                    prob_excess_right: Closed01(0.15),
                };

                NetworkGenome::crossover(parent_left, parent_right, &x, rng)
            }
        }

    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let fitness_evaluator = FitnessEvaluator { target_graph: load_graph("examples/jeffress.gml") };
    println!("{:?}", fitness_evaluator);

    // start with minimal random topology.
    let mut env = Environment::new();

    let template_genome = env.generate_genome(INPUTS, OUTPUTS);

    println!("{:#?}", template_genome);

    let mut initial_pop = Population::<_, Unrated>::new();

    for _ in 0..POP_SIZE {
        // Add a single link gene! This is required, otherwise we can't determine
        // correctly an InnovationRange.
        let genome = env.mutate_add_connection(&template_genome, &mut rng).unwrap();

        initial_pop.add_genome(Box::new(genome));
    }
    assert!(initial_pop.len() == POP_SIZE);

    let mut mater = Mater {
        w_mutate_weight: 0,
        w_mutate_structure: 30,
        w_crossover: 70,
        env: &mut env,
    };

    let compatibility = NetworkGenomeDistance {
        l: LinkGeneListDistance {
            excess: 1.0,
            disjoint: 1.0,
            weight: 0.0,
        },
    };

    let mut runner = Runner {
        pop_size: POP_SIZE,
        elite_percentage: Closed01(0.05),
        selection_percentage: Closed01(0.2),
        tournament_k: 3,
        compatibility_threshold: 1.0,
        compatibility: &compatibility,
        mate: &mut mater,
        fitness: &|genome| Fitness::new(fitness_evaluator.fitness(genome) as f64),
        _marker: PhantomData,
    };

    let (iter, new_pop) = runner.run(initial_pop, &|iter, pop| {
        iter >= 100 || pop.max_fitness().unwrap().get() > 0.99
    }, &mut rng);

    let new_pop = new_pop.sort();

    println!("iter: {}", iter);
    println!("{:#?}", new_pop);
}
