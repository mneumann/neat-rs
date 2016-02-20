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

pub fn load_graph<N, F>(graph_file: &str, convert_node_from_str: F) -> OwnedGraph<N>
    where N: NodeType,
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
