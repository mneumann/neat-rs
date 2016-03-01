extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;
extern crate asexp;
#[macro_use] extern crate log;
extern crate env_logger;

extern crate criterion_stats;

mod common;
mod config;

//use criterion_stats::univariate::Sample;
use neat::population::{Population, Unrated, NicheRunner};
use neat::traits::{FitnessEval};
use neat::genomes::acyclic_network::{Genome, GlobalCache, GlobalInnovationCache, Mater, ElementStrategy};
use neat::fitness::Fitness;
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::Rng;
use std::marker::PhantomData;
use common::{load_graph, Neuron, convert_neuron_from_str, GraphSimilarity, NodeCount, write_gml};
use neat::weight::{Weight, WeightRange};
use closed01::Closed01;

fn genome_to_graph(genome: &Genome<Neuron>) -> OwnedGraph<Neuron> {
    let mut builder = GraphBuilder::new();

    genome.visit_nodes(|ext_id, node_type| {
        // make sure the node exists, even if there are no connection to it.
        let _ = builder.add_node(ext_id, node_type);
    });

    genome.visit_active_links(|src_id, target_id, weight| {
        builder.add_edge(src_id, target_id, Closed01::new(weight.into()));
    });

    return builder.graph();
}

#[derive(Debug)]
struct FitnessEvaluator {
    sim: GraphSimilarity,
}

impl FitnessEval<Genome<Neuron>> for FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Neuron>) -> Fitness {
        Fitness::new(self.sim.fitness(&genome_to_graph(genome)) as f64)
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

    let mut mater = Mater {
        p_crossover: cfg.p_crossover(),
        p_crossover_detail: cfg.probabilistic_crossover(),
        p_mutate_element: cfg.p_mutate_element(),
        weight_perturbance: cfg.weight_perturbance(),
        mutate_weights: cfg.mutate_method_weighting(),
        global_cache: &mut cache,
        element_strategy: &ES,
        _n: PhantomData,
    };

    let mut niche_runner = NicheRunner::new(&fitness_evaluator);

    let niche_size = cfg.population_size() / cfg.num_niches(); 

    for _ in 0..cfg.num_niches() {
        let mut initial_pop = Population::<_, Unrated>::new();

        for _ in 0..niche_size {
            initial_pop.add_genome(Box::new(template_genome.clone()));
        }

        niche_runner.add_unrated_population_as_niche(initial_pop);
    }

    while niche_runner.has_next_iteration(cfg.stop_after_iters()) {
        println!("iteration: {}", niche_runner.current_iteration());

        let best_fitness = niche_runner.best_individual().fitness().get();;
        println!("best fitness: {:2}", best_fitness); 
        println!("num individuals: {}", niche_runner.num_individuals());

        if best_fitness > cfg.stop_if_fitness_better_than() {
            println!("Premature abort.");
            break;
        }

        //let samples = niche_runner.inter_niche_compatibility_distance(100, cfg.genome_compatibility(), &mut rng);
        //println!("samples: {:?}", samples);


        //let samples = Sample::new(&samples);
        //println!("mean: {:?}", samples.mean());
        //println!("median_abs_dev: {:?}", samples.median_abs_dev(None));
        //println!("median_abs_dev_pct: {:?}", samples.median_abs_dev_pct());
        //println!("std_dev: {:?}", samples.std_dev(None));
        //println!("std_dev_pct: {:?}", samples.std_dev_pct());
        //println!("var: {:?}", samples.var(None));

        let threshold = cfg.compatibility_threshold();

        // partition into n niches.
        // niche_runner.partition_n_sorted(cfg.num_niches(), cfg.genome_compatibility(), &mut rng);
        // println!("partitioned into num niches: {}", niche_runner.num_niches());
        /*
        niche_runner.reproduce_global(cfg.population_size(),
                                      cfg.elite_percentage(),
                                      cfg.selection_percentage(),
                                      &mut mater,
                                      &mut rng);
                                      */

        println!("num niches: {}", niche_runner.num_niches());

        niche_runner.reproduce_niche_locally(cfg.population_size(),
                                      cfg.elite_percentage(),
                                      cfg.selection_percentage(),
                                      &mut mater,
                                      &mut rng);

        // If niches do not improve t=10 timesteps, redistribute them to
        // other niches.
        let redistributes = niche_runner.redistribute_niches_with_no_improvement(0.01, 10, 
                                                             cfg.num_niches(), // XXX: rename to max_num_niches()
                                                             threshold,
                                                             cfg.genome_compatibility(),
                                                             &mut rng);
        if redistributes > 0 {
            println!("{} niches redistributed", redistributes);
        }
    }

    let final_pop = niche_runner.into_population().sort();

    {
        let best = final_pop.best_individual().unwrap();
        println!("best fitness: {:.3}", best.fitness().get());
        write_gml("best.gml", &genome_to_graph(best.genome()));
    }

    for (i, ind) in final_pop.into_iter().enumerate() {
        //println!("individual #{}: {:.3}", i, ind.fitness().get());
        write_gml(&format!("ind_{:03}_{}.gml", i, (ind.fitness().get() * 100.0) as usize), &genome_to_graph(ind.genome()));
    }

    write_gml("target.gml", &fitness_evaluator.sim.target_graph);
}
