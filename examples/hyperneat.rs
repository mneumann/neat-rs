extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;
extern crate cppn;

mod common;

use neat::population::{Population, Unrated, Runner};
use neat::genomes::acyclic_network::{Genome, Environment, ElementStrategy};
use neat::fitness::Fitness;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::{Rng, Closed01};
use std::marker::PhantomData;
use common::{load_graph, Mater, Neuron, NodeColors, convert_neuron_from_str};
use cppn::cppn::{Cppn, CppnNodeType};
use cppn::bipolar::BipolarActivationFunction;
use cppn::substrate::Substrate;
use cppn::position::Position2d;

type Node = CppnNodeType<BipolarActivationFunction>;

#[derive(Debug)]
struct FitnessEvaluator {
    target_graph: OwnedGraph<Neuron>,
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Node>) -> f32 {
        let mut substrate = Substrate::new();
        substrate.add_node(Position2d::new(-1.0, -1.0), Neuron::Input);
        substrate.add_node(Position2d::new(1.0, -1.0), Neuron::Input);
        substrate.add_node(Position2d::new(-1.0, 1.0), Neuron::Output);
        substrate.add_node(Position2d::new(0.0, 1.0), Neuron::Output);
        substrate.add_node(Position2d::new(1.0, 1.0), Neuron::Output);

        let mut cppn = Cppn::new(genome.network());

        // now develop the cppn. the result is a graph
        let mut builder = GraphBuilder::new();
        for (i, node) in substrate.nodes().iter().enumerate() {
            let _ = builder.add_node(i, node.data.clone());
        }
        for link in substrate.iter_links(&mut cppn, None) {
            if link.weight >= 0.0 && link.weight <= 1.0 {
                builder.add_edge(link.source_idx,
                                 link.target_idx,
                                 closed01::Closed01::new(link.weight as f32));
            }
        }
        let graph = builder.graph();
        // println!("graph: {:#?}", graph);

        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, NodeColors);
        s.iterate(50, 0.01);
        s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get()
    }
}

struct ES;

impl ElementStrategy<Node> for ES {
    fn full_link_weight() -> f64 {
        1.0
    }

    fn random_link_weight<R: Rng>(rng: &mut R) -> f64 {
        // XXX Choose a weight between -1 and 1?
        rng.gen()
    }

    fn random_node_type<R: Rng>(rng: &mut R) -> Node {
        let af = &[BipolarActivationFunction::Identity,
                   BipolarActivationFunction::Linear,
                   BipolarActivationFunction::Gaussian,
                   BipolarActivationFunction::Sigmoid,
                   BipolarActivationFunction::Sine];

        CppnNodeType::Hidden(*rng.choose(af).unwrap())
    }
}

const POP_SIZE: usize = 100;

fn main() {
    let mut rng = rand::thread_rng();

    let fitness_evaluator = FitnessEvaluator {
        target_graph: load_graph("examples/jeffress.gml", convert_neuron_from_str),
    };

    println!("{:?}", fitness_evaluator);

    // start with minimal random topology.
    let mut env: Environment<Node, ES> = Environment::new();

    let template_genome = {
        let mut genome = Genome::new();

        // 4 inputs (x1,y1,x2,y2)
        for _ in 0..4 {
            env.add_node_to_genome(&mut genome, CppnNodeType::Input);
        }

        // 1 output (y)
        env.add_node_to_genome(&mut genome, CppnNodeType::Output);

        // 1 bias node
        env.add_node_to_genome(&mut genome, CppnNodeType::Bias);

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
        p_crossover_detail: common::default_probabilistic_crossover(),
        mutate_weights: common::default_mutate_weights(),
        env: &mut env,
    };

    let mut runner = Runner {
        pop_size: POP_SIZE,
        elite_percentage: Closed01(0.05),
        selection_percentage: Closed01(0.2),
        tournament_k: 3,
        compatibility_threshold: 1.0,
        compatibility: &common::default_genome_compatibility(),
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
