use std::fs::File;
use std::io::Read;
use std::fmt::Debug;
use graph_neighbor_matching::NodeColorMatching;
use graph_neighbor_matching::graph::OwnedGraph;
use neat::genomes::acyclic_network::NodeType;
use graph_io_gml::parse_gml;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm};

use closed01;

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
                          &|edge_sexp| -> Option<closed01::Closed01<f32>> {
                              edge_sexp.and_then(|se| se.get_float().map(|s| closed01::Closed01::new(s as f32))).
                                  or(Some(closed01::Closed01::zero()))
                          })
                    .unwrap();
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
                           -> closed01::Closed01<f32> {

        // Treat nodes as equal regardless of their activation function or input/output number.
        let eq = match (node_i_value, node_j_value) {
            (&Neuron::Input, &Neuron::Input) => true,
            (&Neuron::Output, &Neuron::Output) => true,
            (&Neuron::Hidden, &Neuron::Hidden) => true,
            _ => false,
        };

        if eq {
            closed01::Closed01::one()
        } else {
            closed01::Closed01::zero()
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
