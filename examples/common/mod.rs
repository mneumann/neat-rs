use std::fs::File;
use std::io::Read;
use std::fmt::Debug;
use graph_neighbor_matching::NodeColorMatching;
use graph_neighbor_matching::graph::OwnedGraph;
use neat::genomes::acyclic_network::NodeType;
use graph_io_gml::parse_gml;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm};
use asexp::Sexp;
use petgraph::{Directed, Graph};
use std::f32::{INFINITY, NEG_INFINITY};
use closed01::Closed01;

fn convert_weight(w: Option<&Sexp>) -> Option<f32> {
    match w {
        Some(s) => s.get_float().map(|f| f as f32),
        None => {
            // use a default
            Some(0.0)
        }
    }
}

fn determine_edge_value_range<T>(g: &Graph<T, f32, Directed>) -> (f32, f32) {
    let mut w_min = INFINITY;
    let mut w_max = NEG_INFINITY;
    for i in g.raw_edges() {
        w_min = w_min.min(i.weight);
        w_max = w_max.max(i.weight);
    }
    (w_min, w_max)
}

fn normalize_to_closed01(w: f32, range: (f32, f32)) -> Closed01<f32> {
    assert!(range.1 >= range.0);
    let dist = range.1 - range.0;
    if dist == 0.0 {
        Closed01::zero()
    } else {
        Closed01::new((w - range.0) / dist)
    }
}

pub fn load_graph<N, F>(graph_file: &str, convert_node_from_str: F) -> OwnedGraph<N>
    where N: Clone + Debug,
          F: Fn(&str) -> N
{
    info!("Loading graph file: {}", graph_file);

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = parse_gml(&graph_s,
                          &|node_sexp| -> Option<N> {
                              node_sexp.and_then(|se| se.get_str().map(|s| convert_node_from_str(s)))
                          },
                          &convert_weight)
                    .unwrap();
    let edge_range = determine_edge_value_range(&graph);
    let graph = graph.map(|_, nw| nw.clone(),
                          |_, &ew| normalize_to_closed01(ew, edge_range));

    OwnedGraph::from_petgraph(&graph)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Neuron {
    Input,
    Output,
    Hidden,
}

impl NodeType for Neuron {
    fn accept_incoming_links(&self) -> bool {
        match *self {
            Neuron::Input => false,
            _ => true,
        }
    }
    fn accept_outgoing_links(&self) -> bool {
        match *self {
            Neuron::Output => false,
            _ => true,
        }
    }
}

pub fn convert_neuron_from_str(s: &str) -> Neuron {
    match s {
        "input" => Neuron::Input,
        "output" => Neuron::Output,
        "hidden" => Neuron::Hidden,
        _ => panic!("Invalid node type/weight"),
    }
}

#[derive(Debug)]
pub struct NodeColors;

impl NodeColorMatching<Neuron> for NodeColors {
    fn node_color_matching(&self,
                           node_i_value: &Neuron,
                           node_j_value: &Neuron)
                           -> Closed01<f32> {

        // Treat nodes as equal regardless of their activation function or input/output number.
        let eq = match (node_i_value, node_j_value) {
            (&Neuron::Input, &Neuron::Input) => true,
            (&Neuron::Output, &Neuron::Output) => true,
            (&Neuron::Hidden, &Neuron::Hidden) => true,
            _ => false,
        };

        if eq {
            Closed01::one()
        } else {
            Closed01::zero()
        }
    }
}

#[derive(Debug)]
pub struct GraphSimilarity {
    pub target_graph: OwnedGraph<Neuron>,
    pub edge_score: bool,
    pub iters: usize,
    pub eps: f32,
}

impl GraphSimilarity {
    // A larger fitness means "better"
    pub fn fitness(&self, graph: OwnedGraph<Neuron>) -> f32 {
        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, NodeColors);
        s.iterate(self.iters, self.eps);
        let assignment = s.optimal_node_assignment();
        if self.edge_score {
            s.score_outgoing_edge_weights_sum_norm(&assignment, ScoreNorm::MaxDegree).get()
        } else {
            s.score_optimal_sum_norm(Some(&assignment), ScoreNorm::MaxDegree).get()
        }
    }
}
