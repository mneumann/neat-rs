extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml as gml;
extern crate closed01;

use neat::population::{Population, Unrated, Runner};
use neat::network::{NetworkGenome, NetworkGenomeDistance, LinkGeneListDistance, NodeType, Environment};
use neat::fitness::Fitness;
use neat::traits::Mate;
use neat::crossover::ProbabilisticCrossover;
use neat::prob::is_probable;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, NodeColorMatching};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::{Rng, Closed01};
use rand::distributions::{WeightedChoice, Weighted, IndependentSample};
use std::marker::PhantomData;

fn load_graph(graph_file: &str) -> OwnedGraph<NodeType> {
    use std::fs::File;
    use std::io::Read;

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = gml::parse_gml(&graph_s,
                               &|sexp| -> Option<NodeType> {
                                   sexp.and_then(|se| se.get_str().map(|s|
                                        match s {
                                            "input" => NodeType::Input,
                                            "output" => NodeType::Output,
                                            "hidden" => NodeType::Hidden,
                                            _ => panic!("Invalid node type/weight"),
                                        }
                                   ))
                               },
                               &|_| -> Option<()> { Some(()) })
                    .unwrap();
    OwnedGraph::from_petgraph(&graph)
}

fn genome_to_graph(genome: &NetworkGenome) -> OwnedGraph<NodeType> {
    let mut builder = GraphBuilder::new();

    for (&innov, node) in genome.node_genes.map.iter() {
        // make sure the node exists, even if there are no connection to it.
        let _ = builder.add_node(innov.get(), node.node_type);
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
    target_graph: OwnedGraph<NodeType>,
}

#[derive(Debug)]
struct NodeColors;

impl NodeColorMatching<NodeType> for NodeColors {
    fn node_color_matching(&self, node_i_value: &NodeType, node_j_value: &NodeType) -> closed01::Closed01<f32> {
        if node_i_value == node_j_value {
            closed01::Closed01::one()
        } else {
            closed01::Closed01::zero()
        }
    }
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &NetworkGenome) -> f32 {
        let graph = genome_to_graph(genome);
        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, NodeColors);
        s.iterate(50, 0.01);
        s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get()
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
