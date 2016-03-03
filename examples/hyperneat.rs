extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;
extern crate cppn;
extern crate asexp;
#[macro_use]
extern crate log;
extern crate env_logger;

mod common;
mod config;

use neat::population::{UnratedPopulation, Individual, Population, PopulationWithRank};
use neat::niching::NicheRunner;
use neat::traits::FitnessEval;
use neat::genomes::acyclic_network::{Genome, GlobalCache, GlobalInnovationCache, Mater,
                                     ElementStrategy};
use neat::fitness::Fitness;
use graph_neighbor_matching::graph::{GraphBuilder, OwnedGraph};
use graph_neighbor_matching::NodeColorWeight;
use rand::Rng;
use std::marker::PhantomData;
use common::{load_graph, Neuron, convert_neuron_from_str, GraphSimilarity, NodeCount, write_gml, NodeLabel, genome_to_petgraph, write_gml_petgraph};
use cppn::cppn::{Cppn, CppnNode};
use cppn::activation_function::GeometricActivationFunction;
use cppn::substrate::Substrate;
use cppn::position::Position2d;
use neat::weight::{Weight, WeightRange};
use neat::distribute::DistributeInterval;
use closed01::Closed01;

type Node = CppnNode<GeometricActivationFunction>;

impl NodeLabel for Node {
    fn node_label(&self) -> Option<String> {
        Some(format!("{:?}", self))
    }
}

fn generate_substrate(node_count: &NodeCount) -> Substrate<Position2d, Neuron> {
    let mut substrate = Substrate::new();

    // XXX: Todo 3d position
    let mut y_iter = DistributeInterval::new(3, -1.0, 1.0); // 3 layers (Input, Hidden, Output)

    // Inputs
    {
        let y = y_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.inputs, -1.0, 1.0) {
            substrate.add_node(Position2d::new(x, y), Neuron::Input);
        }
    }

    // Hidden
    {
        let y = y_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.hidden, -1.0, 1.0) {
            substrate.add_node(Position2d::new(x, y), Neuron::Hidden);
        }
    }

    // Outputs
    {
        let y = y_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.outputs, -1.0, 1.0) {
            substrate.add_node(Position2d::new(x, y), Neuron::Output);
        }
    }

    return substrate;
}


#[derive(Debug)]
struct FitnessEvaluator {
    sim: GraphSimilarity,
    node_count: NodeCount,
    //weight_threshold: f64,
}

impl FitnessEvaluator {
    fn genome_to_graph(&self, genome: &Genome<Node>) -> OwnedGraph<Neuron> {
        let substrate = generate_substrate(&self.node_count);
        let mut cppn = Cppn::new(genome.network());

        // now develop the cppn. the result is a graph
        let mut builder = GraphBuilder::new();
        for (i, node) in substrate.nodes().iter().enumerate() {
            let _ = builder.add_node(i, node.node_type.clone());
        }
        for link in substrate.iter_links(&mut cppn, None) {
            let mut w = link.weight;
            if w > 0.1 {//cppn_weight_threshold_min && w < cppn_weight_threshold_max {
                //let normalized = (w - cppn_weight_threshold_min) / (cppn_weight_threshold_max - cppn_weight_threshold_min);
                //if w > 1.0 {
                //    w = 1.0;
                //}

                builder.add_edge(link.source_idx,
                                 link.target_idx,
                                 Closed01::new(w as f32));
                //}
            }
        }
        return builder.graph();
    }
}

impl FitnessEval<Genome<Node>> for FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Node>) -> Fitness {
        Fitness::new(self.sim.fitness(&self.genome_to_graph(genome)) as f64)
    }
}

struct ES {
    activation_functions: Vec<GeometricActivationFunction>,
    link_weight_range: WeightRange,
    full_link_weight: Weight,
}

impl ElementStrategy<Node> for ES {
    fn link_weight_range(&self) -> WeightRange {
        self.link_weight_range
    }

    fn full_link_weight(&self) -> Weight {
        self.full_link_weight
    }

    fn random_node_type<R: Rng>(&self, rng: &mut R) -> Node {
        CppnNode::hidden(*rng.choose(&self.activation_functions).unwrap())
    }
}

fn main() {
    env_logger::init().unwrap();

    let mut rng = rand::thread_rng();

    let cfg = config::Configuration::from_file();

    println!("{:?}", cfg);

    let target_graph = load_graph(&cfg.target_graph_file(), convert_neuron_from_str);
    let node_count = NodeCount::from_graph(&target_graph);

    let fitness_evaluator = FitnessEvaluator {
        sim: GraphSimilarity {
            target_graph: target_graph,
            edge_score: cfg.edge_score(),
            iters: cfg.neighbormatching_iters(),
            eps: cfg.neighbormatching_eps(),
        },
        node_count: node_count,
    };

    let mut cache = GlobalInnovationCache::new();


    // start with minimal random topology.

    let template_genome = {
        let mut genome = Genome::new();

        // 4 inputs (x1,y1,x2,y2)
        for _ in 0..4 {
            genome.add_node(cache.create_node_innovation(), CppnNode::input(GeometricActivationFunction::Linear));
        }

        // 1 input for distance from source to target neuron
        genome.add_node(cache.create_node_innovation(), CppnNode::input(GeometricActivationFunction::Linear));

        // 1 output (y)
        genome.add_node(cache.create_node_innovation(), CppnNode::output(GeometricActivationFunction::BipolarSigmoid));

        // 1 bias node
        genome.add_node(cache.create_node_innovation(), CppnNode::bias(GeometricActivationFunction::Constant1));

        genome
    };

    let mut niche_runner = NicheRunner::new(&fitness_evaluator);

    for _ in 0..cfg.num_niches() {
        let mut pop = UnratedPopulation::new();

        // XXX: probabilistic round
        let niche_size = cfg.population_size() / cfg.num_niches(); 

        for _ in 0..niche_size {
            pop.add_unrated_individual(Individual::new_unrated(Box::new(template_genome.clone())));
        }

        niche_runner.add_unrated_population_as_niche(pop);
    }

    let es = ES {
        activation_functions: vec![
            //GeometricActivationFunction::Linear,
            GeometricActivationFunction::LinearBipolarClipped,
            GeometricActivationFunction::BipolarGaussian,
            //GeometricActivationFunction::Gaussian,
            GeometricActivationFunction::BipolarSigmoid,
            //GeometricActivationFunction::Absolute,
            GeometricActivationFunction::Sine,
            //GeometricActivationFunction::Cosine
        ],
        link_weight_range: cfg.link_weight_range(),
        full_link_weight: cfg.full_link_weight(),
    };

    let mut mater = Mater {
        p_crossover: cfg.p_crossover(),
        p_crossover_detail_nodes: cfg.probabilistic_crossover_nodes(),
        p_crossover_detail_links: cfg.probabilistic_crossover_links(),
        p_mutate_element: cfg.p_mutate_element(),
        weight_perturbance: cfg.weight_perturbance(),
        mutate_weights: cfg.mutate_method_weighting(),
        global_cache: &mut cache,
        element_strategy: &es,
        _n: PhantomData,
    };

    let mut fitness_log: Vec<f64> = Vec::new();

    while niche_runner.has_next_iteration(cfg.stop_after_iters()) {
        println!("iteration: {}", niche_runner.current_iteration());

        let best_fitness = niche_runner.best_individual().unwrap().fitness().get();;
        println!("best fitness: {:2}", best_fitness); 
        println!("num individuals: {}", niche_runner.num_individuals());

        if best_fitness > cfg.stop_if_fitness_better_than() {
            println!("Premature abort.");
            break;
        }

        let mut top_n_niches = cfg.num_niches();

        niche_runner.reproduce(cfg.population_size(),
                               top_n_niches,
                               cfg.elite_percentage(),
                               cfg.selection_percentage(),
                               cfg.compatibility_threshold(),
                               cfg.genome_compatibility(),
                               &mut mater,
                               &mut rng);
    }

    let final_pop = niche_runner.into_ranked_population();

    {
        let best = final_pop.best_individual().unwrap();
        write_gml("best.gml", &fitness_evaluator.genome_to_graph(best.genome()), &Neuron::node_color_weight);

        // display CPPN
        write_gml_petgraph("best_cppn.gml", &genome_to_petgraph(best.genome()), &|_| 0.0);
    }

    for (i, ind) in final_pop.individuals().iter().enumerate() {
        println!("individual #{}: {:.3}", i, ind.fitness().get());
        write_gml(&format!("ind_{:03}_{}.gml", i, (ind.fitness().get() * 100.0) as usize), &fitness_evaluator.genome_to_graph(ind.genome()), &Neuron::node_color_weight);
    }

    write_gml("target.gml", &fitness_evaluator.sim.target_graph, &Neuron::node_color_weight);
}
