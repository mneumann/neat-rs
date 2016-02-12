extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml as gml;

use neat::population::{Population, Rated, Unrated, Individual};
use neat::network::{NetworkGenome, NetworkGenomeDistance, LinkGeneListDistance, LinkGene, NodeGene, NodeType};
use neat::innovation::{Innovation, InnovationContainer};
use neat::fitness::Fitness;
use neat::traits::Mate;
use neat::crossover::ProbabilisticCrossover;
use neat::is_probable;
use graph_neighbor_matching::similarity_max_degree;
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use std::collections::BTreeMap;
use rand::{Rng, Closed01};
use rand::distributions::{WeightedChoice, Weighted, IndependentSample};

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
        similarity_max_degree(&graph, &self.target_graph, 20, 0.05).get()
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

    fn mutate_add_connection<R: Rng>(&mut self, genome: &NetworkGenome, rng: &mut R) -> Option<NetworkGenome> { 
        genome.find_unconnected_pair(rng).map(|(src, target)| {
            println!("unconnected: {:?} {:?}", src, target);
            // Add this link to the newly created genome
            let new_link_gene = LinkGene {
                source_node_gene: src,
                target_node_gene: target,
                weight: 0.0, // XXX: choose random weight!
                active: true,
            };
            let mut offspring = genome.clone();
            let new_link_innovation = self.new_link_innovation(src, target);
            offspring.link_genes.insert(new_link_innovation, new_link_gene);
            offspring
        })
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
                    let offspring = self.env.mutate_add_connection(parent_left, rng).unwrap_or_else(|| parent_left.clone());
                    offspring
                } else {
                    // AddNode (split existing connection)
                    parent_left.clone()
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

    let rated = initial_pop.rate_par(&|genome| Fitness::new(fitness_evaluator.fitness(genome) as f64));

    let compatibility = NetworkGenomeDistance {
        l: LinkGeneListDistance {
            excess: 1.0,
            disjoint: 1.0,
            weight: 0.0,
        },
    };

    println!("{:#?}", rated);

    let mut mater = Mater {
        w_mutate_weight: 0,
        w_mutate_structure: 30,
        w_crossover: 70,
        env: &mut env,
    };

    let (mut new_rated, new_unrated) = rated.produce_offspring(POP_SIZE,
                                                           Closed01(0.05),
                                                           Closed01(0.2),
                                                           3,
                                                           0.1, // threshold
                                                           &compatibility,
                                                           &mut mater,
                                                           &mut rng);
    let rated = new_unrated.rate_par(&|genome| Fitness::new(fitness_evaluator.fitness(genome) as f64));

    new_rated.merge(rated, None);

    println!("{:#?}", new_rated);
}
