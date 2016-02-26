extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;
extern crate cppn;
extern crate asexp;
#[macro_use] extern crate log;
//extern crate toml;

mod common;
mod config;

use neat::population::{Population, Unrated, Runner};
use neat::genomes::acyclic_network::{Genome, GlobalCache, GlobalInnovationCache, Mater, ElementStrategy};
use neat::fitness::Fitness;
use graph_neighbor_matching::graph::GraphBuilder;
use rand::Rng;
use std::marker::PhantomData;
use common::{load_graph, Neuron, convert_neuron_from_str, GraphSimilarity};
use cppn::cppn::{Cppn, CppnNode};
use cppn::bipolar::BipolarActivationFunction;
use cppn::substrate::Substrate;
use cppn::position::Position2d;
use neat::weight::{Weight, WeightRange};

type Node = CppnNode<BipolarActivationFunction>;

#[derive(Debug)]
struct FitnessEvaluator {
    sim: GraphSimilarity,
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

        self.sim.fitness(graph)
    }
}

struct ES;

impl ElementStrategy<Node> for ES {
    fn link_weight_range(&self) -> WeightRange {
        WeightRange::bipolar(1.0)
    }

    fn full_link_weight(&self) -> Weight {
        WeightRange::bipolar(1.0).high()
    }

    fn random_node_type<R: Rng>(&self, rng: &mut R) -> Node {
        let af = &[BipolarActivationFunction::Identity,
                   BipolarActivationFunction::Linear,
                   BipolarActivationFunction::Gaussian,
                   BipolarActivationFunction::Sigmoid,
                   BipolarActivationFunction::Sine];

        CppnNode::Hidden(*rng.choose(af).unwrap())
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let cfg = config::Configuration::new();

    let target_graph_file = cfg.target_graph_file();
    println!("Using target graph: {}", target_graph_file);

    let fitness_evaluator = FitnessEvaluator {
        sim: GraphSimilarity {
            target_graph: load_graph(&cfg.target_graph_file(), convert_neuron_from_str),
            edge_score: cfg.edge_score(),
            iters: cfg.neighbormatching_iters(),
            eps: cfg.neighbormatching_eps()
        }
    };

    let mut cache = GlobalInnovationCache::new();


    // start with minimal random topology.

    let template_genome = {
        let mut genome = Genome::new();

        // 4 inputs (x1,y1,x2,y2)
        for _ in 0..4 {
            genome.add_node(cache.create_node_innovation(), CppnNode::Input);
        }

        // 1 output (y)
        genome.add_node(cache.create_node_innovation(), CppnNode::Output);

        // 1 bias node
        genome.add_node(cache.create_node_innovation(), CppnNode::Bias);

        genome
    };

    let mut initial_pop = Population::<_, Unrated>::new();

    for _ in 0..cfg.population_size() {
        initial_pop.add_genome(Box::new(template_genome.clone()));
    }
    assert!(initial_pop.len() == cfg.population_size());

    let mut mater = Mater {
        p_crossover: cfg.p_crossover(),
        p_crossover_detail: cfg.probabilistic_crossover(),
        p_mutate_element: cfg.p_mutate_element(),
        weight_perturbance: cfg.weight_perturbance(),
        mutate_weights: cfg.mutate_weights(),
        global_cache: &mut cache,
        element_strategy: &ES,
        _n: PhantomData,
    };

    let mut runner = Runner {
        pop_size: cfg.population_size(),
        elite_percentage: cfg.elite_percentage(),
        selection_percentage: cfg.selection_percentage(),
        compatibility_threshold: cfg.compatibility_threshold(),
        compatibility: &cfg.genome_compatibility(),
        mate: &mut mater,
        fitness: &|genome| Fitness::new(fitness_evaluator.fitness(genome) as f64),
        _marker: PhantomData,
    };

    let max_iters = cfg.stop_after_iters();
    let good_fitness = cfg.stop_if_fitness_better_than();
    let (iter, new_pop) = runner.run(initial_pop,
                                     &|iter, pop| {
                                         iter >= max_iters || pop.best_individual().unwrap().fitness().get() > good_fitness
                                     },
                                     &mut rng);

    println!("iter: {}", iter);
    println!("{:#?}", new_pop.best_individual());
}
