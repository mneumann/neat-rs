use innovation::{Innovation, InnovationRange};
use acyclic_network::{Network, NodeIndex};
pub use acyclic_network::NodeType;
use traits::{Distance, Genotype};
use weight::Weight;
use alignment_metric::AlignmentMetric;
use std::collections::BTreeMap;
use alignment::{Alignment, align_sorted_iterators};

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

struct CombinedAlignmentMetric {
    node_metric: AlignmentMetric,
    link_metric: AlignmentMetric,
}

impl CombinedAlignmentMetric {
    fn new() -> Self {
        CombinedAlignmentMetric {
            node_metric: AlignmentMetric::new(),
            link_metric: AlignmentMetric::new(),
        }
    }
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

    /// Counts the number of matching, disjoint and excess node innovation numbers between `self`
    /// and `other`.

    fn node_alignment_metric(&self, other: &Self) -> AlignmentMetric {
        let left = self.node_innovation_map.keys();
        let right = other.node_innovation_map.keys();

        let mut node_metric = AlignmentMetric::new();
        align_sorted_iterators(left, right, Ord::cmp, |alignment| {
            match alignment {
                Alignment::Match(_l, _r) => {
                    node_metric.matching += 1;
                }
                ref align if align.is_disjoint() => {
                    node_metric.disjoint += 1;
                }
                ref align if align.is_excess() => {
                    node_metric.excess += 1;
                }
                _ => unreachable!()
            }
        });

        node_metric
    }

    /// Counts the number of matching, disjoint and excess link AND node innovation numbers between
    /// `self` and `right`.

    fn combined_alignment_metric(&self, other: &Self) -> CombinedAlignmentMetric {
        let mut metric = CombinedAlignmentMetric::new();

        let left = self.node_innovation_map.iter();
        let right = other.node_innovation_map.iter();

        let left_link_innov_range = self.link_innovation_range();
        let right_link_innov_range = other.link_innovation_range();


        let left_network = &self.network;
        let right_network = &other.network;

        align_sorted_iterators(left, right, |&(kl, _), &(kr, _)| Ord::cmp(kl, kr), |node_alignment| {
            match node_alignment {
                Alignment::Match((_, &left_node_index), (_, &right_node_index)) => {
                    metric.node_metric.matching += 1;

                    // Both nodes are topological identical. So the link innovations can
                    // also match up.
                    align_sorted_iterators(left_network.link_iter_for_node(left_node_index),
                                           right_network.link_iter_for_node(right_node_index),
                                           |&(_, left_link), &(_, right_link)| 
                                               Ord::cmp(&left_link.external_link_id(),
                                                        &right_link.external_link_id()),
                                            |link_alignment| {

                                                if let Alignment::Match((_, left_link), (_, right_link)) = link_alignment {
                                                    // we have a link match!
                                                    metric.link_metric.matching += 1;

                                                    // add up the weight distance
                                                    metric.link_metric.weight_distance += (left_link.weight().0 - right_link.weight().0).abs();
                                                } else if link_alignment.is_disjoint() {
                                                    // the link is locally disjoint (list of links of the node) 
                                                    metric.link_metric.disjoint += 1;
                                                } else if link_alignment.is_left() {
                                                    let &(_, left_link) = link_alignment.get_left().unwrap();

                                                    if right_link_innov_range.contains(&LinkInnovation(left_link.external_link_id().0)) {
                                                        metric.link_metric.disjoint += 1;
                                                    } else {
                                                        metric.link_metric.excess += 1;
                                                    }
                                                } else if link_alignment.is_right() {
                                                    let &(_, right_link) = link_alignment.get_right().unwrap();

                                                    if left_link_innov_range.contains(&LinkInnovation(right_link.external_link_id().0)) {
                                                        metric.link_metric.disjoint += 1;
                                                    } else {
                                                        metric.link_metric.excess += 1;
                                                    }
                                                } else {
                                                    unreachable!();
                                                }
                                            });
                }

                // in general, if a node is disjoint (or excess), it's link innovations cannot match up! 

                ref align_left if align_left.is_left() => {

                    let &(_, &left_node_index) = align_left.get_left().unwrap();
                    // XXX: Optimize: once we hit an excess link id, all remaining ids are excess as well.
                    for (_, left_link) in left_network.link_iter_for_node(left_node_index) {
                        // check if link is disjoint or excess
                        if right_link_innov_range.contains(&LinkInnovation(left_link.external_link_id().0)) {
                            metric.link_metric.disjoint += 1;
                        } else {
                            metric.link_metric.excess += 1;
                        }
                    }

                    if align_left.is_excess() {
                        metric.node_metric.excess += 1;
                    } else if align_left.is_disjoint() {
                        metric.node_metric.disjoint += 1;
                    } else {
                        unreachable!();
                    }

                }

                ref align_right if align_right.is_right() => {

                    let &(_, &right_node_index) = align_right.get_right().unwrap();
                    // XXX: Optimize: once we hit an excess link id, all remaining ids are excess as well.
                    for (_, right_link) in right_network.link_iter_for_node(right_node_index) {
                        // check if link is disjoint or excess
                        if left_link_innov_range.contains(&LinkInnovation(right_link.external_link_id().0)) {
                            metric.link_metric.disjoint += 1;
                        } else {
                            metric.link_metric.excess += 1;
                        }
                    }

                    if align_right.is_excess() {
                        metric.node_metric.excess += 1;
                    } else if align_right.is_disjoint() {
                        metric.node_metric.disjoint += 1;
                    } else {
                        unreachable!();
                    }

                }

                _ => {
                    unreachable!()
                }

            }
        });

        metric
    }

    /// Determine the genetic compatibility between `self` and `other` in terms of matching,
    /// disjoint and excess genes, as well as weight distance.
    ///
    /// The first thing which we have to do is to determine the range of innovations of each
    /// genome, i.e. it's min and max values.

    fn alignment_metric(&self, other: &Self) -> AlignmentMetric {
        let mut metric = AlignmentMetric::new();

        let left_link_innov_range = self.link_innovation_range();
        let right_link_innov_range = self.link_innovation_range();

        // let left_node_innov_range = self.node_innovation_range();
        // let right_node_innov_range = 
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

    #[test]
    fn test_node_align_metric() {
        let mut left = Genome::<NT>::new();
        let mut right = Genome::<NT>::new();

        let m = left.node_alignment_metric(&right);
        assert_eq!(0, m.matching);
        assert_eq!(0, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(5), NT);
        let m = left.node_alignment_metric(&right);
        assert_eq!(0, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(10), NT);
        let m = left.node_alignment_metric(&right);
        assert_eq!(0, m.matching);
        assert_eq!(2, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(6), NT);
        let m = left.node_alignment_metric(&right);
        assert_eq!(0, m.matching);
        assert_eq!(2, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(5), NT);
        let m = left.node_alignment_metric(&right);
        assert_eq!(1, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(6), NT);
        let m = left.node_alignment_metric(&right);
        assert_eq!(2, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(11), NT);
        let m = left.node_alignment_metric(&right);
        assert_eq!(2, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);
    }

    #[test]
    fn test_combined_align_metric() {
        let mut left = Genome::<NT>::new();
        let mut right = Genome::<NT>::new();

        assert_eq!(left.node_alignment_metric(&right),
                   left.combined_alignment_metric(&right).node_metric);

        left.add_node(NodeInnovation(5), NT);
        assert_eq!(left.node_alignment_metric(&right),
                   left.combined_alignment_metric(&right).node_metric);

        left.add_node(NodeInnovation(10), NT);
        assert_eq!(left.node_alignment_metric(&right),
                   left.combined_alignment_metric(&right).node_metric);

        right.add_node(NodeInnovation(6), NT);
        assert_eq!(left.node_alignment_metric(&right),
                   left.combined_alignment_metric(&right).node_metric);

        right.add_node(NodeInnovation(5), NT);
        assert_eq!(left.node_alignment_metric(&right),
                   left.combined_alignment_metric(&right).node_metric);

        left.add_node(NodeInnovation(6), NT);
        assert_eq!(left.node_alignment_metric(&right),
                   left.combined_alignment_metric(&right).node_metric);

        right.add_node(NodeInnovation(11), NT);
        assert_eq!(left.node_alignment_metric(&right),
                   left.combined_alignment_metric(&right).node_metric);
    }

}
