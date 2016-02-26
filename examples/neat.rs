extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;
extern crate asexp;
#[macro_use] extern crate log;

mod common;
mod config;

use neat::population::{Population, Unrated, Runner};
use neat::genomes::acyclic_network::{Genome, GlobalCache, GlobalInnovationCache, Mater, ElementStrategy};
use neat::fitness::Fitness;
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::Rng;
use std::marker::PhantomData;
use common::{load_graph, Neuron, convert_neuron_from_str, GraphSimilarity, NodeCount, write_gml};
use neat::weight::{Weight, WeightRange};

fn genome_to_graph(genome: &Genome<Neuron>) -> OwnedGraph<Neuron> {
    let mut builder = GraphBuilder::new();

    genome.visit_nodes(|ext_id, node_type| {
        // make sure the node exists, even if there are no connection to it.
        let _ = builder.add_node(ext_id, node_type);
    });

    genome.visit_active_links(|src_id, target_id, weight| {
        builder.add_edge(src_id, target_id, closed01::Closed01::new(weight.into()));
    });

    return builder.graph();
}

#[derive(Debug)]
struct FitnessEvaluator {
    sim: GraphSimilarity,
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Neuron>) -> f32 {
        self.sim.fitness(genome_to_graph(genome))
    }
}

struct ES;

impl ElementStrategy<Neuron> for ES {
    fn link_weight_range(&self) -> WeightRange {
        WeightRange::unipolar(1.0)
    }

    fn full_link_weight(&self) -> Weight {
        WeightRange::unipolar(1.0).high()
    }

    fn random_node_type<R: Rng>(&self, _rng: &mut R) -> Neuron {
        Neuron::Hidden
    }
}

fn main() {
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
            eps: cfg.neighbormatching_eps()
        }
    };

    let mut cache = GlobalInnovationCache::new();

    // start with minimal random topology.
    //
    // Generates a template Genome with `n_inputs` input nodes and `n_outputs` output nodes.
    // The genome will not have any link nodes.

    let template_genome = {
        let mut genome = Genome::new();
        assert!(node_count.inputs > 0 && node_count.outputs > 0);

        for _ in 0..node_count.inputs {
            genome.add_node(cache.create_node_innovation(), Neuron::Input);
        }
        for _ in 0..node_count.outputs {
            genome.add_node(cache.create_node_innovation(), Neuron::Output);
        }
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
    let best = new_pop.best_individual().unwrap();
    println!("{:#?}", best);

    info!("Writing best.gml");
    write_gml("best.gml", &genome_to_graph(best.genome()));
}
