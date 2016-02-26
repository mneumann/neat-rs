use std::env;
use neat::crossover::ProbabilisticCrossover;
use neat::mutate::MutateMethodWeighting;
use neat::weight::{WeightRange, WeightPerturbanceMethod};
use neat::prob::Prob;
use neat::genomes::acyclic_network::GenomeDistance;
use rand::Closed01;
use asexp::Sexp;

#[derive(Debug)]
pub struct Configuration {
    edge_score: bool
}

fn parse_bool(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        _ => panic!("invalid bool"),
    }
}

impl Configuration {
    pub fn new() -> Self {
        Configuration {
            edge_score: true
        }
    }

    /// Treats the strings as top-level `asexp` file.
    ///
    /// # Panics
    ///
    /// If the string has the wrong format or mandantory options are missing. 

    pub fn from_str(s: &str) -> Self {
        let expr = Sexp::parse_toplevel(s).unwrap();
        let map = expr.into_map().unwrap();
        Configuration {
            edge_score: parse_bool(map["edge_score"].get_str().unwrap())
        }
    }

    pub fn p_crossover(&self) -> Prob {
        Prob::new(0.5)
    }

    pub fn p_mutate_element(&self) -> Prob {
        // 1% mutation rate per link
        Prob::new(0.01)
    }

    pub fn weight_perturbance(&self) -> WeightPerturbanceMethod {
        WeightPerturbanceMethod::JiggleUniform{range: WeightRange::bipolar(0.1)}
    }

    pub fn elite_percentage(&self) -> Closed01<f64> {
        Closed01(0.05)
    }

    pub fn selection_percentage(&self) -> Closed01<f64> {
        Closed01(0.2)
    }

    pub fn compatibility_threshold(&self) -> f64 {
        1.0
    }

    pub fn stop_after_iters(&self) -> usize {
        100
    }

    pub fn stop_if_fitness_better_than(&self) -> f64 {
        0.99
    }

    pub fn neighbormatching_iters(&self) -> usize {
        50
    }

    pub fn neighbormatching_eps(&self) -> f32 {
        0.01
    }

    pub fn edge_score(&self) -> bool {
        self.edge_score
    }

    pub fn population_size(&self) -> usize {
        100
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
