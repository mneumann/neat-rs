use std::fs::File;
use std::io::Read;
use graph_neighbor_matching::graph::OwnedGraph;
use neat::genomes::acyclic_network::{NodeType, Genome, Environment, ElementStrategy};
use neat::crossover::ProbabilisticCrossover;
use graph_io_gml::parse_gml;
use rand::{Rng, Closed01};
use neat::mutate::{MutateMethod, MutateMethodWeighting};
use neat::traits::Mate;
use neat::prob::is_probable;
use std::fmt::Debug;
use graph_neighbor_matching::NodeColorMatching;
use closed01;

pub fn load_graph<N, F>(graph_file: &str, convert_node_from_str: F) -> OwnedGraph<N>
    where N: Clone + Debug,
          F: Fn(&str) -> N
{

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = parse_gml(&graph_s,
                          &|sexp| -> Option<N> {
                              sexp.and_then(|se| se.get_str().map(|s| convert_node_from_str(s)))
                          },
                          &|_| -> Option<()> { Some(()) })
                    .unwrap();
    OwnedGraph::from_petgraph(&graph)
}

pub struct Mater<'a, N, S>
    where N: NodeType + 'a,
          S: ElementStrategy<N> + 'a
{
    // probability for crossover. P_mutate = 1.0 - p_crossover
    pub p_crossover: Closed01<f32>,
    pub p_crossover_detail: ProbabilisticCrossover,
    pub mutate_weights: MutateMethodWeighting,
    pub env: &'a mut Environment<N, S>,
}

impl<'a, N, S> Mate<Genome<N>> for Mater<'a, N, S>
    where N: NodeType + 'a,
          S: ElementStrategy<N> + 'a
{
    // Add an argument that descibes whether both genomes are of equal fitness.
    // Pass individual, which includes the fitness.
    fn mate<R: Rng>(&mut self,
                    parent_left: &Genome<N>,
                    parent_right: &Genome<N>,
                    prefer_mutate: bool,
                    rng: &mut R)
                    -> Genome<N> {
        if prefer_mutate == false && is_probable(&self.p_crossover, rng) {
            Genome::crossover(parent_left, parent_right, &self.p_crossover_detail, rng)
        } else {
            // mutate
            let mutate_method = MutateMethod::random_with(&self.mutate_weights, rng);
            self.env
                .mutate(parent_left, mutate_method, rng)
                .or_else(|| self.env.mutate(parent_right, mutate_method, rng))
                .unwrap_or_else(|| parent_left.clone())
        }
    }
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
