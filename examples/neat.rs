extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;

mod common;

use neat::population::{Population, Unrated, Runner};
use neat::genomes::acyclic_network::{NodeType, Genome, GenomeDistance, Environment,
                                     ElementStrategy};
use neat::fitness::Fitness;
use neat::crossover::ProbabilisticCrossover;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, NodeColorMatching};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::{Rng, Closed01};
use std::marker::PhantomData;
use neat::mutate::MutateMethodWeighting;
use neat::gene::Gene;
use common::{load_graph, Mater};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Node {
    Input,
    Output,
    Hidden,
}

impl NodeType for Node {
    fn accept_incoming_links(&self) -> bool {
        match *self {
            Node::Input => false,
            _ => true,
        }
    }
    fn accept_outgoing_links(&self) -> bool {
        match *self {
            Node::Output => false,
            _ => true,
        }
    }
}

fn convert_node_from_str(s: &str) -> Node {
    match s {
        "input" => Node::Input,
        "output" => Node::Output,
        "hidden" => Node::Hidden,
        _ => panic!("Invalid node type/weight"),
    }
}

fn genome_to_graph(genome: &Genome<Node>) -> OwnedGraph<Node> {
    let mut builder = GraphBuilder::new();

    genome.visit_node_genes(|node_gene| {
        // make sure the node exists, even if there are no connection to it.
        let _ = builder.add_node(node_gene.innovation().get(), node_gene.node_type);
    });

    genome.visit_active_link_genes(|link_gene| {
        builder.add_edge(link_gene.source_node_gene.get(),
                         link_gene.target_node_gene.get(),
                         closed01::Closed01::new(link_gene.weight as f32));

    });

    return builder.graph();
}

#[derive(Debug)]
struct NodeColors;

impl NodeColorMatching<Node> for NodeColors {
    fn node_color_matching(&self,
                           node_i_value: &Node,
                           node_j_value: &Node)
                           -> closed01::Closed01<f32> {

        // Treat nodes as equal regardless of their activation function or input/output number.
        let eq = match (node_i_value, node_j_value) {
            (&Node::Input, &Node::Input) => true,
            (&Node::Output, &Node::Output) => true,
            (&Node::Hidden, &Node::Hidden) => true,
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
    target_graph: OwnedGraph<Node>,
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Node>) -> f32 {
        let graph = genome_to_graph(genome);
        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, NodeColors);
        s.iterate(50, 0.01);
        s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get()
    }
}

struct ES;

impl ElementStrategy<Node> for ES {
    fn random_link_weight<R: Rng>(rng: &mut R) -> f64 {
        // XXX Choose a weight between -1 and 1?
        rng.gen()
    }
    fn random_node_type<R: Rng>(_rng: &mut R) -> Node {
        Node::Hidden
    }
}

const POP_SIZE: usize = 100;
const INPUTS: usize = 2;
const OUTPUTS: usize = 3;


fn main() {
    let mut rng = rand::thread_rng();

    let fitness_evaluator = FitnessEvaluator {
        target_graph: load_graph("examples/jeffress.gml", convert_node_from_str),
    };

    println!("{:?}", fitness_evaluator);

    // start with minimal random topology.
    let mut env: Environment<Node, ES> = Environment::new();

    // Generates a template Genome with `n_inputs` input nodes and `n_outputs` output nodes.
    // The genome will not have any link nodes.

    let template_genome = {
        let mut genome = Genome::new();
        assert!(INPUTS > 0 && OUTPUTS > 0);

        for _ in 0..INPUTS {
            env.add_node_to_genome(&mut genome, Node::Input);
        }
        for _ in 0..OUTPUTS {
            env.add_node_to_genome(&mut genome, Node::Output);
        }
        genome
    };

    println!("{:#?}", template_genome);

    let mut initial_pop = Population::<_, Unrated>::new();

    for _ in 0..POP_SIZE {
        initial_pop.add_genome(Box::new(template_genome.clone()));
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

    let compatibility = GenomeDistance {
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