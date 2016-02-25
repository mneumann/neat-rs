extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;

mod common;

use neat::population::{Population, Unrated, Runner};
use neat::genomes::acyclic_network::{Genome, GlobalCache, GlobalInnovationCache};
use neat::fitness::Fitness;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::{Rng, Closed01};
use std::marker::PhantomData;
use common::{load_graph, Mater, Neuron, NodeColors, convert_neuron_from_str, ElementStrategy};
use neat::weight::{Weight, WeightRange, WeightPerturbanceMethod};
use neat::prob::Prob;

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
    target_graph: OwnedGraph<Neuron>,
    edge_score: bool,
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Neuron>) -> f32 {
        let graph = genome_to_graph(genome);
        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, NodeColors);
        s.iterate(50, 0.01);
        let assignment = s.optimal_node_assignment();
        if self.edge_score {
            s.score_outgoing_edge_weights_sum_norm(&assignment, ScoreNorm::MaxDegree).get()
        } else {
            s.score_optimal_sum_norm(Some(&assignment), ScoreNorm::MaxDegree).get()
        }
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

const POP_SIZE: usize = 100;
const INPUTS: usize = 2;
const OUTPUTS: usize = 3;

fn main() {
    let mut rng = rand::thread_rng();

    let fitness_evaluator = FitnessEvaluator {
        target_graph: load_graph("examples/jeffress.gml", convert_neuron_from_str),
        edge_score: true,
    };

    let mut cache = GlobalInnovationCache::new();

    // start with minimal random topology.
    //
    // Generates a template Genome with `n_inputs` input nodes and `n_outputs` output nodes.
    // The genome will not have any link nodes.

    let template_genome = {
        let mut genome = Genome::new();
        assert!(INPUTS > 0 && OUTPUTS > 0);

        for _ in 0..INPUTS {
            genome.add_node(cache.create_node_innovation(), Neuron::Input);
        }
        for _ in 0..OUTPUTS {
            genome.add_node(cache.create_node_innovation(), Neuron::Output);
        }
        genome
    };

    let mut initial_pop = Population::<_, Unrated>::new();

    for _ in 0..POP_SIZE {
        initial_pop.add_genome(Box::new(template_genome.clone()));
    }
    assert!(initial_pop.len() == POP_SIZE);

    let mut mater = Mater {
        p_crossover: Prob::new(0.5),
        p_crossover_detail: common::default_probabilistic_crossover(),
        p_mutate_element: Prob::new(0.01), // 1% mutation rate per link
        weight_perturbance: WeightPerturbanceMethod::JiggleUniform{range: WeightRange::bipolar(0.2)},
        mutate_weights: common::default_mutate_weights(),
        global_cache: &mut cache,
        element_strategy: &ES,
        _n: PhantomData,
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
                                         iter >= 100 || pop.best_individual().unwrap().fitness().get() > 0.99
                                     },
                                     &mut rng);

    //let new_pop = new_pop.sort();

    println!("iter: {}", iter);
    println!("{:#?}", new_pop.best_individual());
}
