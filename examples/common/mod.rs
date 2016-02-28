use std::fs::File;
use std::io::Read;
use std::fmt::Debug;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, WeightedNodeColors, NodeColorWeight};
use graph_neighbor_matching::graph::OwnedGraph;
use graph_neighbor_matching::{Graph, Edges};
use neat::genomes::acyclic_network::NodeType;
use graph_io_gml::parse_gml;
use asexp::Sexp;
use petgraph::Directed;
use petgraph::Graph as PetGraph;
use std::f32::{INFINITY, NEG_INFINITY};
use closed01::Closed01;
use std::io::{self, Write};

fn convert_weight(w: Option<&Sexp>) -> Option<f32> {
    match w {
        Some(s) => s.get_float().map(|f| f as f32),
        None => {
            // use a default
            Some(0.0)
        }
    }
}

fn determine_edge_value_range<T>(g: &PetGraph<T, f32, Directed>) -> (f32, f32) {
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

pub fn write_gml(filename: &str, graph: &OwnedGraph<Neuron>) {
    let mut file = File::create(filename).unwrap();
    to_gml(&mut file, graph).unwrap();
}

pub fn to_gml<W: Write>(wr: &mut W, graph: &OwnedGraph<Neuron>) -> io::Result<()> {
    try!(writeln!(wr, "graph ["));
    try!(writeln!(wr, "  directed 1"));

    for nidx in 0..graph.num_nodes() {
        let node_type: f32 = graph.node_value(nidx).node_color_weight();
        try!(writeln!(wr, "  node [id {} weight {:.1}]", nidx, node_type));
    }
    for nidx in 0..graph.num_nodes() {
        let edges = graph.out_edges_of(nidx);
        for eidx in 0..edges.num_edges() {
            try!(writeln!(wr, "  edge [source {} target {} weight {:.2}]", nidx, edges.nth_edge(eidx).unwrap(), edges.nth_edge_weight(eidx).unwrap().get()));
        }
    }
    try!(writeln!(wr, "]"));
    Ok(())
}

#[derive(Debug)]
pub struct NodeCount {
    pub inputs: usize,
    pub outputs: usize,
    pub hidden: usize,
}

impl NodeCount {
    pub fn from_graph(graph: &OwnedGraph<Neuron>) -> Self {
        let mut cnt = NodeCount {
            inputs: 0,
            outputs: 0,
            hidden: 0
        };

        for node in graph.nodes() {
            match node.node_value() {
                &Neuron::Input => {
                    cnt.inputs += 1;
                }
                &Neuron::Output => {
                    cnt.outputs += 1;
                }
                &Neuron::Hidden => {
                    cnt.hidden += 1;
                }
            }
        }

        return cnt;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Neuron {
    Input,
    Output,
    Hidden,
}

impl NodeColorWeight for Neuron {
    fn node_color_weight(&self) -> f32 {
        match *self { 
            Neuron::Input => 0.0,
            Neuron::Hidden => 1.0,
            Neuron::Output => 2.0,
        }
    }
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
pub struct GraphSimilarity {
    pub target_graph: OwnedGraph<Neuron>,
    pub edge_score: bool,
    pub iters: usize,
    pub eps: f32,
}

impl GraphSimilarity {
    // A larger fitness means "better"
    pub fn fitness(&self, graph: &OwnedGraph<Neuron>) -> f32 {
        let mut s = SimilarityMatrix::new(graph, &self.target_graph, WeightedNodeColors);
        s.iterate(self.iters, self.eps);
        let assignment = s.optimal_node_assignment();
        let score = s.score_optimal_sum_norm(Some(&assignment), ScoreNorm::MaxDegree).get();
        if self.edge_score {
            score * s.score_outgoing_edge_weights_sum_norm(&assignment, ScoreNorm::MaxDegree).get()
        } else {
            score
        }
    }
}
