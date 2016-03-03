extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;
extern crate asexp;
#[macro_use] extern crate log;
extern crate env_logger;

//extern crate criterion_stats;

mod common;
mod config;

//use criterion_stats::univariate::Sample;
use neat::population::{UnratedPopulation, Individual, Population, PopulationWithRank};
use neat::niching::NicheRunner;

use neat::traits::{FitnessEval};
use neat::genomes::acyclic_network::{Genome, GlobalCache, GlobalInnovationCache, Mater, ElementStrategy};
use neat::fitness::Fitness;
use rand::Rng;
use std::marker::PhantomData;
use common::{load_graph, Neuron, convert_neuron_from_str, GraphSimilarity, NodeCount, write_gml, genome_to_network};
use neat::weight::{Weight, WeightRange};
use graph_neighbor_matching::NodeColorWeight;

#[derive(Debug)]
struct FitnessEvaluator {
    sim: GraphSimilarity,
}

impl FitnessEval<Genome<Neuron>> for FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Neuron>) -> Fitness {
        Fitness::new(self.sim.fitness(&genome_to_network(genome)) as f64)
    }
}

struct ES {
    link_weight_range: WeightRange,
    full_link_weight: Weight,
}

impl ElementStrategy<Neuron> for ES {
    fn link_weight_range(&self) -> WeightRange {
        self.link_weight_range
    }

    fn full_link_weight(&self) -> Weight {
        self.full_link_weight
    }

    fn random_node_type<R: Rng>(&self, _rng: &mut R) -> Neuron {
        Neuron::Hidden
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

    let es = ES {
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

        fitness_log.push(best_fitness);

        let mut top_n_niches = cfg.num_niches();

        /*
        if fitness_log.len() >= 10 {
            let previous_fitness = fitness_log[fitness_log.len() - 10];
            if best_fitness - previous_fitness < 0.01 {
                // no change within 10 timesteps
                top_n_niches = 2;
            }
        }
        */

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
        println!("best fitness: {:.3}", best.fitness().get());
        write_gml("best.gml", &genome_to_network(best.genome()), &Neuron::node_color_weight);
    }

    for (i, ind) in final_pop.individuals().iter().enumerate() {
        //println!("individual #{}: {:.3}", i, ind.fitness().get());
        write_gml(&format!("ind_{:03}_{}.gml", i, (ind.fitness().get() * 100.0) as usize), &genome_to_network(ind.genome()), &Neuron::node_color_weight);
    }

    write_gml("target.gml", &fitness_evaluator.sim.target_graph, &Neuron::node_color_weight);
}
