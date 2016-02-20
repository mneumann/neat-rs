use std::str::FromStr;
use std::fs::File;
use std::io::Read;
use graph_neighbor_matching::graph::OwnedGraph;
use neat::genomes::acyclic_network::NodeType;
use graph_io_gml::parse_gml;

pub fn load_graph<N>(graph_file: &str) -> OwnedGraph<N>
    where N: NodeType + FromStr<Err=&'static str>,
{

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = parse_gml(&graph_s,
                               &|sexp| -> Option<N> {
                                   sexp.and_then(|se| se.get_str().map(|s| {
                                       match N::from_str(s) {
                                           Ok(n) => n,
                                           Err(err) => panic!(err),
                                       }
                                   }))
                               },
                               &|_| -> Option<()> { Some(()) })
                    .unwrap();
    OwnedGraph::from_petgraph(&graph)
}
