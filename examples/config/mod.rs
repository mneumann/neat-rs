use std::env;
use neat::crossover::ProbabilisticCrossover;
use neat::mutate::MutateMethodWeighting;
use neat::prob::Prob;
use neat::genomes::acyclic_network::GenomeDistance;

pub struct Configuration;

impl Configuration {
    pub fn new() -> Self {
        Configuration
    }

    pub fn target_graph_file(&self) -> String {
        env::args().nth(1).unwrap().to_owned()
    }

    pub fn genome_compatibility(&self) -> GenomeDistance {
        GenomeDistance {
            excess: 1.0,
            disjoint: 1.0,
            weight: 0.0,
        }
    }

    pub fn probabilistic_crossover(&self) -> ProbabilisticCrossover {
        ProbabilisticCrossover {
            prob_match_left: Prob::new(0.5), // NEAT always selects a random parent for matching genes
            prob_disjoint_left: Prob::new(0.9),
            prob_excess_left: Prob::new(0.9),
            prob_disjoint_right: Prob::new(0.15),
            prob_excess_right: Prob::new(0.15),
        }
    }

    pub fn mutate_weights(&self) -> MutateMethodWeighting {
        // XXX:
        MutateMethodWeighting {
            w_modify_weight: 100,
            w_add_node: 1,
            w_add_connection: 10,
            w_delete_connection: 1,
        }
    }
}
