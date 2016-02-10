extern crate neat;

use neat::population::UnratedPopulation;
use neat::network::{NetworkGenome, NodeGene, NodeType};
use neat::innovation::{Innovation, InnovationContainer};


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
