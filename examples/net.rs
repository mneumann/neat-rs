extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml as gml;
extern crate closed01;

use neat::population::{Population, Unrated, Runner};
use neat::network::{NetworkGenome, NetworkGenomeDistance, NodeType, Environment,
                    LinkWeightStrategy};
use neat::fitness::Fitness;
use neat::traits::Mate;
use neat::crossover::ProbabilisticCrossover;
use neat::prob::is_probable;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, NodeColorMatching};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::{Rng, Closed01};
use std::marker::PhantomData;
use neat::mutate::{MutateMethod, MutateMethodWeighting};

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
                                   sexp.and_then(|se| {
                                       se.get_str().map(|s| {
                                           match s {
                                               "input" => NodeType::Input,
                                               "output" => NodeType::Output,
                                               "hidden" => NodeType::Hidden,
                                               _ => panic!("Invalid node type/weight"),
                                           }
                                       })
                                   })
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
    fn node_color_matching(&self,
                           node_i_value: &NodeType,
                           node_j_value: &NodeType)
                           -> closed01::Closed01<f32> {
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

#[derive(Debug)]
struct LinkWeightS;

impl LinkWeightStrategy for LinkWeightS {}

const POP_SIZE: usize = 100;
const INPUTS: usize = 2;
const OUTPUTS: usize = 3;

struct Mater<'a, T: LinkWeightStrategy + 'a> {
    // probability for crossover. P_mutate = 1.0 - p_crossover
    p_crossover: Closed01<f32>,
    p_crossover_detail: ProbabilisticCrossover,
    mutate_weights: MutateMethodWeighting,
    env: &'a mut Environment<T>,
}

impl<'a, T: LinkWeightStrategy> Mate<NetworkGenome> for Mater<'a, T> {
    // Add an argument that descibes whether both genomes are of equal fitness.
    // Pass individual, which includes the fitness.
    fn mate<R: Rng>(&mut self,
                    parent_left: &NetworkGenome,
                    parent_right: &NetworkGenome,
                    prefer_mutate: bool,
                    rng: &mut R)
                    -> NetworkGenome {
        if prefer_mutate == false && is_probable(&self.p_crossover, rng) {
            NetworkGenome::crossover(parent_left, parent_right, &self.p_crossover_detail, rng)
        } else {
            // mutate
            let mutate_method = MutateMethod::random_with(&self.mutate_weights, rng);
            self.env.mutate(parent_left, mutate_method, rng).
                or_else(|| self.env.mutate(parent_right, mutate_method, rng)).
                unwrap_or_else(|| parent_left.clone())
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let fitness_evaluator = FitnessEvaluator { target_graph: load_graph("examples/jeffress.gml") };
    println!("{:?}", fitness_evaluator);

    // start with minimal random topology.
    let mut env: Environment<LinkWeightS> = Environment::new();

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
        p_crossover: Closed01(0.5),
        p_crossover_detail: ProbabilisticCrossover {
                prob_match_left: Closed01(0.5), /* NEAT always selects a random parent for matching genes */
                prob_disjoint_left: Closed01(0.9),
                prob_excess_left: Closed01(0.9),
                prob_disjoint_right: Closed01(0.15),
                prob_excess_right: Closed01(0.15),
            },
        mutate_weights: MutateMethodWeighting {
            w_modify_weight: 1,
            w_add_node: 1,
            w_add_connection: 1,
        },

        env: &mut env,
    };

    let compatibility = NetworkGenomeDistance {
        excess: 1.0,
        disjoint: 1.0,
        weight: 0.0,
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

    let (iter, new_pop) = runner.run(initial_pop,
                                     &|iter, pop| {
                                         iter >= 100 || pop.max_fitness().unwrap().get() > 0.99
                                     },
                                     &mut rng);

    let new_pop = new_pop.sort();

    println!("iter: {}", iter);
    println!("{:#?}", new_pop);
}
