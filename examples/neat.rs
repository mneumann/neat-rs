extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;

mod common;

use neat::population::{Population, Unrated, Runner};
use neat::genomes::acyclic_network::{Genome, Environment, ElementStrategy};
use neat::fitness::Fitness;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use rand::{Rng, Closed01};
use std::marker::PhantomData;
use neat::gene::Gene;
use common::{load_graph, Mater, Neuron, NodeColors, convert_neuron_from_str};

fn genome_to_graph(genome: &Genome<Neuron>) -> OwnedGraph<Neuron> {
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
struct FitnessEvaluator {
    target_graph: OwnedGraph<Neuron>,
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Neuron>) -> f32 {
        let graph = genome_to_graph(genome);
        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, NodeColors);
        s.iterate(50, 0.01);
        s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get()
    }
}

struct ES;

impl ElementStrategy<Neuron> for ES {
    fn full_link_weight() -> f64 {
        1.0
    }
    fn random_link_weight<R: Rng>(rng: &mut R) -> f64 {
        // XXX Choose a weight between -1 and 1?
        rng.gen()
    }
    fn random_node_type<R: Rng>(_rng: &mut R) -> Neuron {
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
    };

    println!("{:?}", fitness_evaluator);

    // start with minimal random topology.
    let mut env: Environment<Neuron, ES> = Environment::new();

    // Generates a template Genome with `n_inputs` input nodes and `n_outputs` output nodes.
    // The genome will not have any link nodes.

    let template_genome = {
        let mut genome = Genome::new();
        assert!(INPUTS > 0 && OUTPUTS > 0);

        for _ in 0..INPUTS {
            env.add_node_to_genome(&mut genome, Neuron::Input);
        }
        for _ in 0..OUTPUTS {
            env.add_node_to_genome(&mut genome, Neuron::Output);
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
