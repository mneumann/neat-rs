extern crate neat;
extern crate graph_neighbor_matching;
extern crate graph_io_gml as gml;

use neat::population::UnratedPopulation;
use neat::network::{NetworkGenome, NodeGene, NodeType};
use neat::innovation::{Innovation, InnovationContainer};
use graph_neighbor_matching::similarity_max_degree;
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

fn load_graph(graph_file: &str) -> OwnedGraph {
    use std::fs::File;
    use std::io::Read;

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = gml::parse_gml(&graph_s, &|_| -> Option<()> {Some(())}, &|_| -> Option<()> {Some(())}).unwrap();
    OwnedGraph::from_petgraph(&graph)
}

fn genome_to_graph(genome: &NetworkGenome) -> OwnedGraph {
    let mut builder = GraphBuilder::new();

    for (&innov, node) in genome.node_genes.map.iter() {
        // make sure the node exists, even if there are no connection to it.
        builder.add_or_replace_node(innov.get())
    }

    for link in genome.link_genes.map.values() {
        if link.active {
            builder.add_edge_unweighted(link.source_node_gene.get(), link.target_node_gene.get());
        }
    }

    return builder.graph();
}

#[derive(Debug)]
struct FitnessEvaluator {
    target_graph: OwnedGraph,
}

impl FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &NetworkGenome) -> f32 {
        let graph = genome_to_graph(genome);
        similarity_max_degree(&graph, &self.target_graph, 20, 0.05).get()
    }
}

struct GlobalContext {
    node_innovation_counter: Innovation,
    link_innovation_counter: Innovation,
}

impl GlobalContext {
    fn new() -> GlobalContext {
        GlobalContext {
            node_innovation_counter: Innovation::new(0),
            link_innovation_counter: Innovation::new(0),
        }
    }

    // Generates a NetworkGenome with `n_inputs` input nodes and `n_outputs` output nodes.
    // The genome will not have any link nodes.
    fn generate_network_genome(&mut self, n_inputs: usize, n_outputs: usize) -> NetworkGenome {
        assert!(n_inputs > 0 && n_outputs > 0);
        let mut nodes = InnovationContainer::new();
        for _ in 0..n_inputs {
            nodes.insert(self.node_innovation_counter.next().unwrap(), NodeGene {
                node_type: NodeType::Input,
            });
        }
        assert!(nodes.len() == n_inputs);
        for _ in 0..n_outputs {
            nodes.insert(self.node_innovation_counter.next().unwrap(), NodeGene {
                node_type: NodeType::Output,
            });
        }
        assert!(nodes.len() == n_inputs + n_outputs);
        NetworkGenome {
            link_genes: InnovationContainer::new(),
            node_genes: nodes,
        }
    }
}

const POP_SIZE: usize = 100;
const INPUTS: usize = 2;
const OUTPUTS: usize = 3;

fn main() {
    let fitness_evaluator = FitnessEvaluator {
        target_graph: load_graph("examples/jeffress.gml"),
    };
    println!("{:?}", fitness_evaluator);

    // start with minimal random topology.
    let mut ctx = GlobalContext::new();

    let template_genome = ctx.generate_network_genome(INPUTS, OUTPUTS);

    println!("{:#?}", template_genome);

    let mut initial_pop = UnratedPopulation::new();

    for _ in 0..POP_SIZE {
        initial_pop.add(Box::new(template_genome.clone()));
    }
    assert!(initial_pop.len() == POP_SIZE);
}
