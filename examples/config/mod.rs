use std::env;
use neat::crossover::ProbabilisticCrossover;
use neat::mutate::MutateMethodWeighting;
use neat::weight::{WeightRange, WeightPerturbanceMethod};
use neat::prob::Prob;
use neat::genomes::acyclic_network::GenomeDistance;
use rand::Closed01;
use asexp::Sexp;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

#[derive(Debug)]
pub struct Configuration {
    population_size: usize,
    edge_score: bool,
    target_graph_file: Option<String>,

    w_modify_weight: u32,
    w_add_node: u32,
    w_add_connection: u32,
    w_delete_connection: u32,
}

fn conv_bool(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        _ => panic!("invalid bool"),
    }
}

fn parse_bool(map: &BTreeMap<String, Sexp>, key: &str) -> Option<bool> {
    if map.contains_key(key) {
        Some(conv_bool(map[key].get_str().unwrap()))
    } else {
        None
    }
}

fn parse_uint(map: &BTreeMap<String, Sexp>, key: &str) -> Option<u64> {
    if map.contains_key(key) {
        Some(map[key].get_uint().unwrap())
    } else {
        None
    }
}

fn parse_string(map: &BTreeMap<String, Sexp>, key: &str) -> Option<String> {
    if map.contains_key(key) {
        Some(map[key].get_str().unwrap().to_owned())
    } else {
        None
    }
}

impl Configuration {

    pub fn from_file() -> Self {
        let filename = env::args().nth(1).unwrap().to_owned();
        let mut data = String::new();
        let _ = File::open(&filename).unwrap().read_to_string(&mut data).unwrap();

        Configuration::from_str(&data)
    }

    pub fn new() -> Self {
        Configuration {
            edge_score: false,
            population_size: 100,
            w_modify_weight: 100,
            w_add_node: 1,
            w_add_connection: 10,
            w_delete_connection: 1,

            target_graph_file: None,
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
        let mut cfg = Configuration::new();

        if let Some(val) = parse_bool(&map, "edge_score") { cfg.edge_score = val; }
        if let Some(val) = parse_uint(&map, "population_size") { cfg.population_size = val as usize; }

        if let Some(val) = parse_uint(&map, "w_modify_weight") { cfg.w_modify_weight = val as u32; }
        if let Some(val) = parse_uint(&map, "w_add_node") { cfg.w_add_node = val as u32; }
        if let Some(val) = parse_uint(&map, "w_add_connection") { cfg.w_add_connection = val as u32; }
        if let Some(val) = parse_uint(&map, "w_delete_connection") { cfg.w_delete_connection = val as u32; }

        if let Some(val) = parse_string(&map, "target_graph_file") { cfg.target_graph_file = Some(val); }

        cfg
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
        self.population_size
    }

    pub fn target_graph_file(&self) -> String {
        if let Some(ref file) = self.target_graph_file {
            file.to_owned()
        } else {
            env::args().nth(2).unwrap().to_owned()
        }
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
        MutateMethodWeighting {
            w_modify_weight: self.w_modify_weight,
            w_add_node: self.w_add_node,
            w_add_connection: self.w_add_connection,
            w_delete_connection: self.w_delete_connection,
        }
    }
}
