use std::fs::File;
use std::io::Read;
use graph_neighbor_matching::graph::OwnedGraph;
use neat::genomes::acyclic_network::{NodeType, Genome, GenomeDistance, GlobalCache};
use neat::crossover::ProbabilisticCrossover;
use graph_io_gml::parse_gml;
use rand::Rng;
use neat::mutate::{MutateMethod, MutateMethodWeighting};
use neat::traits::Mate;
use std::fmt::Debug;
use graph_neighbor_matching::NodeColorMatching;
use closed01;
use neat::weight::{Weight, WeightRange, WeightPerturbanceMethod};
use neat::prob::Prob;
use std::marker::PhantomData;

/// This trait is used to specialize link weight creation and node activation function creation.

pub trait ElementStrategy<NT: NodeType>
{
    fn link_weight_range(&self) -> WeightRange;
    fn full_link_weight(&self) -> Weight;
    fn random_node_type<R: Rng>(&self, rng: &mut R) -> NT;
}

pub fn default_genome_compatibility() -> GenomeDistance {
    GenomeDistance {
        excess: 1.0,
        disjoint: 1.0,
        weight: 0.0,
    }
}

pub fn default_probabilistic_crossover() -> ProbabilisticCrossover {
    ProbabilisticCrossover {
        prob_match_left: Prob::new(0.5), // NEAT always selects a random parent for matching genes
        prob_disjoint_left: Prob::new(0.9),
        prob_excess_left: Prob::new(0.9),
        prob_disjoint_right: Prob::new(0.15),
        prob_excess_right: Prob::new(0.15),
    }
}

pub fn default_mutate_weights() -> MutateMethodWeighting {
    // XXX:
    MutateMethodWeighting {
        w_modify_weight: 100,
        w_add_node: 1,
        w_add_connection: 10,
        w_delete_connection: 1,
    }
}

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

pub struct Mater<'a, N, S, C>
    where N: NodeType + 'a,
          S: ElementStrategy<N> + 'a,
          C: GlobalCache + 'a
{
    // probability for crossover. P_mutate = 1.0 - p_crossover
    pub p_crossover: Prob,
    pub p_crossover_detail: ProbabilisticCrossover,
    pub p_mutate_element: Prob,
    pub weight_perturbance: WeightPerturbanceMethod,
    pub mutate_weights: MutateMethodWeighting,
    pub global_cache: &'a mut C,
    pub element_strategy: &'a S, 
    pub _n: PhantomData<N>, 
}

impl<'a, N, S, C> Mate<Genome<N>> for Mater<'a, N, S, C>
    where N: NodeType + 'a,
          S: ElementStrategy<N> + 'a,
          C: GlobalCache + 'a
{
    // Add an argument that descibes whether both genomes are of equal fitness.
    // Pass individual, which includes the fitness.
    fn mate<R: Rng>(&mut self,
                    parent_left: &Genome<N>,
                    parent_right: &Genome<N>,
                    prefer_mutate: bool,
                    rng: &mut R)
                    -> Genome<N> {
        if prefer_mutate == false && self.p_crossover.flip(rng) {
            Genome::crossover(parent_left, parent_right, &self.p_crossover_detail, rng)
        } else {
            // mutate
            let mut offspring = parent_left.clone();

            let mutate_method = MutateMethod::random_with(&self.mutate_weights, rng);
            match mutate_method {
                MutateMethod::ModifyWeight => {
                    let _modifications = offspring.mutate_link_weights_uniformly(
                        self.p_mutate_element,
                        &self.weight_perturbance,
                        &self.element_strategy.link_weight_range(), rng);
                }
                MutateMethod::AddConnection => {
                    let link_weight = self.element_strategy.link_weight_range().random_weight(rng);
                    let _modified = offspring.mutate_add_link(link_weight, self.global_cache, rng);
                }
                MutateMethod::DeleteConnection => {
                    let _modified = offspring.mutate_delete_link(rng);
                }
                MutateMethod::AddNode => {
                    let second_link_weight = self.element_strategy.full_link_weight();
                    let node_type = self.element_strategy.random_node_type(rng);
                    let _modified = offspring.mutate_add_node(node_type, second_link_weight, self.global_cache, rng);
                }
            }

            return offspring;
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
