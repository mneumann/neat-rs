extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml as gml;
extern crate closed01;
extern crate petgraph;

use neat::population::{Population, Unrated, Runner};
use neat::genomes::network::{NetworkGenome, NetworkGenomeDistance, NodeType, Environment, ElementStrategy};
use neat::fitness::Fitness;
use neat::traits::Mate;
use neat::crossover::ProbabilisticCrossover;
use neat::prob::is_probable;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, NodeColorMatching};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::{Rng, Closed01};
use std::marker::PhantomData;
use std::fmt::Debug;
use neat::mutate::{MutateMethod, MutateMethodWeighting};

fn node_type_from_str(s: &str) -> NodeType {
    match s {
        "input" => NodeType::Input,
        "output" => NodeType::Output,
        "hidden" => NodeType::Hidden { activation_function: 0 }, // XXX
        _ => panic!("Invalid node type/weight"),
    }
}

// NT is the returned node type
fn load_graph<NT, F>(graph_file: &str, node_weight_fn: &F) -> OwnedGraph<NT>
    where F: Fn(&str) -> NT,
          NT: Clone + Debug
{
    use std::fs::File;
    use std::io::Read;

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = gml::parse_gml(&graph_s,
                               &|sexp| -> Option<NT> {
                                   sexp.and_then(|se| se.get_str().map(|s| node_weight_fn(s)))
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
            builder.add_edge(link.source_node_gene.get(),
                             link.target_node_gene.get(),
                             closed01::Closed01::new(link.weight as f32));
        }
    }

    return builder.graph();
}

#[derive(Debug)]
struct NodeColors;

impl NodeColorMatching<NodeType> for NodeColors {
    fn node_color_matching(&self,
                           node_i_value: &NodeType,
                           node_j_value: &NodeType)
                           -> closed01::Closed01<f32> {

        // Treat nodes as equal regardless of their activation function or input/output number.
        let eq = match (node_i_value, node_j_value) {
            (&NodeType::Input, &NodeType::Input) => true,
            (&NodeType::Output, &NodeType::Output) => true,
            (&NodeType::Hidden{..}, &NodeType::Hidden{..}) => true,
            _ => false,
        };

        if eq {
            closed01::Closed01::one()
        } else {
            closed01::Closed01::zero()
        }
    }
}

#[derive(Debug)]
struct FitnessEvaluator {
    target_graph: OwnedGraph<NodeType>,
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

struct ES;

impl ElementStrategy for ES {
    fn random_link_weight<R: Rng>(rng: &mut R) -> f64 {
        // XXX Choose a weight between -1 and 1?
        rng.gen()
    }
    fn random_activation_function<R: Rng>(rng: &mut R) -> u32 {
        rng.gen_range(0, 5)
    }
}

const POP_SIZE: usize = 100;
const INPUTS: usize = 2;
const OUTPUTS: usize = 3;

struct Mater<'a, T: ElementStrategy + 'a> {
    // probability for crossover. P_mutate = 1.0 - p_crossover
    p_crossover: Closed01<f32>,
    p_crossover_detail: ProbabilisticCrossover,
    mutate_weights: MutateMethodWeighting,
    env: &'a mut Environment<T>,
}

impl<'a, T: ElementStrategy> Mate<NetworkGenome> for Mater<'a, T> {
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
            self.env
                .mutate(parent_left, mutate_method, rng)
                .or_else(|| self.env.mutate(parent_right, mutate_method, rng))
                .unwrap_or_else(|| parent_left.clone())
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let fitness_evaluator = FitnessEvaluator {
        target_graph: load_graph("examples/jeffress.gml", &node_type_from_str),
    };

    println!("{:?}", fitness_evaluator);

    // start with minimal random topology.
    let mut env: Environment<ES> = Environment::new();

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
