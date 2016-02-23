use innovation::{Innovation, InnovationRange};
use acyclic_network::{Network, NodeIndex};
pub use acyclic_network::NodeType;
use traits::{Distance, Genotype};
use weight::Weight;
use alignment_metric::AlignmentMetric;
use std::collections::BTreeMap;

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AnyInnovation(usize);

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NodeInnovation(usize);

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LinkInnovation(usize);

impl Innovation for AnyInnovation {
}

impl Innovation for NodeInnovation {
}

impl Innovation for LinkInnovation {
}

/// Genome representing a feed-forward (acyclic) network.
///
/// Each node is uniquely identified by it's Innovation number. Each link is sorted according it's
/// associated Innovation number.
///
/// We have to keep both the `network` and the `node_innovation_map` in sync. That is, whenever we
/// add or remove a node, we have to update both.

#[derive(Clone, Debug)]
pub struct Genome<NT: NodeType> {

    /// Represents the acyclic feed forward network.

    network: Network<NT, Weight, AnyInnovation>,

    /// Maps the external id (innovation number) which is globally allocated, to the internal
    /// network node index.

    node_innovation_map: BTreeMap<NodeInnovation, NodeIndex>,
}

impl<NT: NodeType> Genotype for Genome<NT> {}

impl<NT: NodeType> Genome<NT> {
    fn new() -> Self {
        Genome {
            network: Network::new(),
            node_innovation_map: BTreeMap::new(),
        }
    }

    /// Determine the genetic compatibility between `self` and `other` in terms of matching,
    /// disjoint and excess genes, as well as weight distance.
    ///
    /// The first thing which we have to do is to determine the range of innovations of each
    /// genome, i.e. it's min and max values.

    fn alignment_metric(&self, other: &Self) -> AlignmentMetric {
        let mut metric = AlignmentMetric::new();

        //let min_node_innovation = self. 

        // ...
        // 1. Determine the 

        return metric;
    }

    /// Determine the genomes range of node innovations. If the genome
    /// contains no nodes, this will return `None`. Otherwise it will
    /// return the Some((min, max)).
    ///
    /// # Complexity
    ///
    /// This runs in O(log n).

    fn node_innovation_range(&self) -> InnovationRange<NodeInnovation> {
        let mut range = InnovationRange::empty();

        if let Some(&min) = self.node_innovation_map.keys().min() {
            range.insert(min);
        }
        if let Some(&max) = self.node_innovation_map.keys().max() {
            range.insert(max);
        }

        return range;
    }

    /// Determine the link innovation range for that Genome.
    ///
    /// # Complexity
    ///
    /// O(n) where `n` is the number of nodes.

    fn link_innovation_range(&self) -> InnovationRange<LinkInnovation> {
        let mut range = InnovationRange::empty();

        let network = &self.network;
        network.each_node_with_index(|_, node_idx| {
            if let Some(link) = network.first_link_of_node(node_idx) {
                range.insert(link.external_link_id());
            }
            if let Some(link) = network.last_link_of_node(node_idx) {
                range.insert(link.external_link_id());
            }
        });

        range.map(|i| LinkInnovation(i.0))
    }

    /// Add a link between `source_node` and `target_node`. Associates the new
    /// link with `link_innovation` and gives it `weight`.
    ///
    /// Does not check for cycles. Test for cycles before using this method!
    ///
    /// # Note
    ///
    /// Does not panic or abort if a link with the same link innovation is added.
    ///
    /// # Panics
    ///
    /// If one of `source_node` or `target_node` does not exist.
    ///
    /// If a link between these nodes already exists!
    ///
    /// # Complexity 
    ///
    /// This runs in O(k) + O(log n), where `k` is the number of edges of `source_node`.
    /// This is because we keep the edges sorted. `n` is the number of nodes, because
    /// we have to lookup the internal node indices from the node innovations.

    fn add_link(&mut self, source_node: NodeInnovation, target_node: NodeInnovation,
                link_innovation: LinkInnovation, weight: Weight) {
        let source_node_index = self.node_innovation_map[&source_node];
        let target_node_index = self.node_innovation_map[&target_node];

        debug_assert!(!self.network.link_would_cycle(source_node_index, target_node_index));
        debug_assert!(self.network.valid_link(source_node_index, target_node_index).is_ok());

        let _link_index = self.network.add_link(source_node_index, target_node_index, weight, AnyInnovation(link_innovation.0));
    }

    fn link_count(&self) -> usize {
        self.network.link_count()
    }

    /// Add a new node with external id `node_innovation` and of type `node_type`
    /// to the genome.
    ///
    /// # Panics
    ///
    /// Panics if a node with the same innovation already exists in the genome.

    fn add_node(&mut self, node_innovation: NodeInnovation, node_type: NT) {
        if self.node_innovation_map.contains_key(&node_innovation) {
            panic!("Duplicate node_innovation");
        }

        let node_index = self.network.add_node(node_type, AnyInnovation(node_innovation.0));
        self.node_innovation_map.insert(node_innovation, node_index);
    }

    fn node_count(&self) -> usize {
        assert!(self.node_innovation_map.len() == self.network.node_count()); 
        return self.node_innovation_map.len();
    }
}

#[cfg(test)]
mod tests {
    use super::{NodeType, Genome, NodeInnovation, LinkInnovation};
    use weight::Weight;
    use innovation::InnovationRange;

    #[derive(Clone, Debug)]
    struct NT;
    impl NodeType for NT {
        fn accept_incoming_links(&self) -> bool { true }
        fn accept_outgoing_links(&self) -> bool { true }
    }

    #[test]
    fn test_add_node() {
        let mut genome = Genome::<NT>::new();
        assert_eq!(0, genome.node_count());
        genome.add_node(NodeInnovation(0), NT); 
        assert_eq!(1, genome.node_count());
        genome.add_node(NodeInnovation(1), NT); 
        assert_eq!(2, genome.node_count());
    }

    #[test]
    #[should_panic(expected = "Duplicate node_innovation")]
    fn test_add_duplicate_node() {
        let mut genome = Genome::<NT>::new();
        genome.add_node(NodeInnovation(0), NT); 
        genome.add_node(NodeInnovation(0), NT); 
    }

    #[test]
    fn test_add_link() {
        let mut genome = Genome::<NT>::new();
        let n0 = NodeInnovation(0);
        let n1 = NodeInnovation(1);
        let n2 = NodeInnovation(2);

        genome.add_node(n0, NT); 
        genome.add_node(n1, NT); 
        genome.add_node(n2, NT); 

        assert_eq!(0, genome.link_count());

        genome.add_link(n0, n1, LinkInnovation(0), Weight(0.0));
        assert_eq!(1, genome.link_count());

        genome.add_link(n0, n2, LinkInnovation(0), Weight(0.0));
        assert_eq!(2, genome.link_count());
    }

    #[test]
    fn test_link_innovation_range() {
        let mut genome = Genome::<NT>::new();
        let n0 = NodeInnovation(0);
        let n1 = NodeInnovation(1);
        let n2 = NodeInnovation(2);

        genome.add_node(n0, NT); 
        genome.add_node(n1, NT); 
        genome.add_node(n2, NT); 

        assert_eq!(InnovationRange::Empty, genome.link_innovation_range());

        genome.add_link(n0, n1, LinkInnovation(5), Weight(0.0));
        assert_eq!(InnovationRange::Single(LinkInnovation(5)), genome.link_innovation_range());

        genome.add_link(n0, n2, LinkInnovation(1), Weight(0.0));
        assert_eq!(InnovationRange::FromTo(LinkInnovation(1), LinkInnovation(5)), genome.link_innovation_range());

        genome.add_link(n1, n2, LinkInnovation(99), Weight(0.0));
        assert_eq!(InnovationRange::FromTo(LinkInnovation(1), LinkInnovation(99)), genome.link_innovation_range());
    }

    #[test]
    fn test_node_innovation_range() {
        let mut genome = Genome::<NT>::new();
        assert_eq!(InnovationRange::Empty, genome.node_innovation_range());

        genome.add_node(NodeInnovation(5), NT); 
        assert_eq!(InnovationRange::Single(NodeInnovation(5)), genome.node_innovation_range());

        genome.add_node(NodeInnovation(7), NT); 
        assert_eq!(InnovationRange::FromTo(NodeInnovation(5), NodeInnovation(7)), genome.node_innovation_range());

        genome.add_node(NodeInnovation(6), NT); 
        assert_eq!(InnovationRange::FromTo(NodeInnovation(5), NodeInnovation(7)), genome.node_innovation_range());

        genome.add_node(NodeInnovation(4), NT); 
        assert_eq!(InnovationRange::FromTo(NodeInnovation(4), NodeInnovation(7)), genome.node_innovation_range());

        genome.add_node(NodeInnovation(1), NT); 
        assert_eq!(InnovationRange::FromTo(NodeInnovation(1), NodeInnovation(7)), genome.node_innovation_range());

        genome.add_node(NodeInnovation(1000), NT); 
        assert_eq!(InnovationRange::FromTo(NodeInnovation(1), NodeInnovation(1000)), genome.node_innovation_range());
    }

}
