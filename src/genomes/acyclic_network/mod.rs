use innovation::{Innovation, InnovationRange};
use acyclic_network::{Network, NodeIndex};
pub use acyclic_network::NodeType;
use traits::{Distance, Genotype};
use weight::Weight;
use alignment_metric::AlignmentMetric;
use std::collections::BTreeMap;
use alignment::{Alignment, align_sorted_iterators, LeftOrRight};
use std::cmp;
use rand::{Rng};
use crossover::ProbabilisticCrossover;
use std::convert::Into;

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AnyInnovation(usize);

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeInnovation(usize);

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinkInnovation(usize);

impl Innovation for AnyInnovation {}

impl Into<NodeInnovation> for AnyInnovation {
    fn into(self) -> NodeInnovation {
        NodeInnovation(self.0)
    }
}

impl Into<LinkInnovation> for AnyInnovation {
    fn into(self) -> LinkInnovation {
        LinkInnovation(self.0)
    }
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

#[inline]
fn count_disjoint_or_excess<I: Innovation>(metric: &mut AlignmentMetric, range: &InnovationRange<I>, innovation: I) {
    if range.contains(&innovation) {
        metric.disjoint += 1;
    } else {
        metric.excess += 1;
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

    /// Counts the number of matching, disjoint and excess node innovation numbers between
    /// `left_genome` and `right_genome`.

    fn node_alignment_metric(left_genome: &Self, right_genome: &Self) -> AlignmentMetric {
        let mut node_metric = AlignmentMetric::new();
        node_metric.max_len = cmp::max(left_genome.node_innovation_map.len(), right_genome.node_innovation_map.len());

        let left = left_genome.node_innovation_map.keys();
        let right = right_genome.node_innovation_map.keys();

        align_sorted_iterators(left, right, Ord::cmp, |alignment| {
            match alignment {
                Alignment::Match(_l, _r) => {
                    node_metric.matching += 1;
                }
                Alignment::Disjoint(..) => {
                    node_metric.disjoint += 1;
                }
                Alignment::Excess(..) => {
                    node_metric.excess += 1;
                }
            }
        });

        node_metric
    }

    /// Determine the genetic compatibility between `left_genome` and `right_genome` in terms of matching,
    /// disjoint and excess genes (both node and link genes), as well as weight distance.

    fn combined_alignment_metric(left_genome: &Self, right_genome: &Self) -> CombinedAlignmentMetric {
        let mut metric = CombinedAlignmentMetric::new();
        metric.node_metric.max_len = cmp::max(left_genome.network.node_count(), right_genome.network.node_count());
        metric.link_metric.max_len = cmp::max(left_genome.network.link_count(), right_genome.network.link_count());

        let left_nodes= left_genome.node_innovation_map.iter();
        let right_nodes= right_genome.node_innovation_map.iter();

        let left_link_innov_range = left_genome.link_innovation_range();
        let right_link_innov_range = right_genome.link_innovation_range();

        let left_network = &left_genome.network;
        let right_network = &right_genome.network;

        align_sorted_iterators(left_nodes, right_nodes, |&(kl, _), &(kr, _)| Ord::cmp(kl, kr), |node_alignment| {
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
                                                match link_alignment { 
                                                    Alignment::Match((_, left_link), (_, right_link)) => {
                                                        // we have a link match!
                                                        metric.link_metric.matching += 1;

                                                        // add up the weight distance
                                                        metric.link_metric.weight_distance += (left_link.weight().0 - right_link.weight().0).abs();
                                                    }
                                                    Alignment::Disjoint(..) => {
                                                        // the link is locally disjoint (list of links of the node) 
                                                        metric.link_metric.disjoint += 1;
                                                    }
                                                    Alignment::Excess((_, left_link), LeftOrRight::Left) => {
                                                        count_disjoint_or_excess(&mut metric.link_metric, &right_link_innov_range, left_link.external_link_id().into());
                                                    }
                                                    Alignment::Excess((_, right_link), LeftOrRight::Right) => {
                                                        count_disjoint_or_excess(&mut metric.link_metric, &left_link_innov_range, right_link.external_link_id().into());
                                                    }
                                                }
                                            });
                }

                // in general, if a node is disjoint (or excess), it's link innovations cannot match up! 
                // XXX: Optimize: once we hit an excess link id, all remaining ids are excess as well.

                Alignment::Disjoint((_, &node_index), LeftOrRight::Left) => {
                    metric.node_metric.disjoint += 1;

                    for (_, link) in left_network.link_iter_for_node(node_index) {
                        count_disjoint_or_excess(&mut metric.link_metric, &right_link_innov_range, link.external_link_id().into());
                    }
                }

                Alignment::Disjoint((_, &node_index), LeftOrRight::Right) => {
                    metric.node_metric.disjoint += 1;

                    for (_, link) in right_network.link_iter_for_node(node_index) {
                        count_disjoint_or_excess(&mut metric.link_metric, &left_link_innov_range, link.external_link_id().into());
                    }
                }

                Alignment::Excess((_, &node_index), LeftOrRight::Left) => {
                    metric.node_metric.excess += 1;

                    for (_, link) in left_network.link_iter_for_node(node_index) {
                        count_disjoint_or_excess(&mut metric.link_metric, &right_link_innov_range, link.external_link_id().into());
                    }
                }

                Alignment::Excess((_, &node_index), LeftOrRight::Right) => {
                    metric.node_metric.excess += 1;

                    for (_, link) in right_network.link_iter_for_node(node_index) {
                        count_disjoint_or_excess(&mut metric.link_metric, &left_link_innov_range, link.external_link_id().into());
                    }
                }
            }
        });

        metric
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

    /// Returns a reference to the feed forward network.

    pub fn network(&self) -> &Network<NT, Weight, AnyInnovation> {
        &self.network
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

    pub fn add_link(&mut self, source_node: NodeInnovation, target_node: NodeInnovation,
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

    pub fn add_node(&mut self, node_innovation: NodeInnovation, node_type: NT) {
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

    /// Performs a crossover operation on the two genomes `left_genome` and `right_genome`,
    /// producing a new offspring genome.

    fn crossover<R: Rng>(left_genome: &Self, right_genome: &Self, c: &ProbabilisticCrossover, rng: &mut R) -> Self {
        let mut offspring = Genome::new();

        return offspring;
    }

    /// Crossover the nodes of `left_genome` and `right_genome`. So either take a node from the
    /// left or the right, depending on randomness and `c`.

    fn crossover_nodes<R: Rng>(left_genome: &Self, right_genome: &Self, offspring: &mut Self, c: &ProbabilisticCrossover, rng: &mut R) {
        let left_nodes = left_genome.node_innovation_map.iter();
        let right_nodes = right_genome.node_innovation_map.iter();

        let left_network = &left_genome.network;
        let right_network = &right_genome.network;

        align_sorted_iterators(left_nodes, right_nodes, |&(kl, _), &(kr, _)| Ord::cmp(kl, kr), |node_alignment| {
            match node_alignment {
                Alignment::Match((&ni_l, &left_node_index), (&ni_r, &right_node_index)) => {
                    // Both genomes have the same node gene (node innovation).
                    // Either take the node type from the left genome or the right.

                    debug_assert!(ni_l == ni_r);

                    if c.prob_match_left.flip(rng) {
                        // take from left
                        offspring.add_node(ni_l, left_network.node(left_node_index).node_type().clone());
                    } else {
                        // take from right
                        offspring.add_node(ni_r, right_network.node(right_node_index).node_type().clone());
                    }
                }

                Alignment::Disjoint((&ni_l, &left_node_index), LeftOrRight::Left) => {
                    if c.prob_disjoint_left.flip(rng) {
                        offspring.add_node(ni_l, left_network.node(left_node_index).node_type().clone());
                    }
                }

                Alignment::Disjoint((&ni_r, &right_node_index), LeftOrRight::Right) => {
                    if c.prob_disjoint_right.flip(rng) {
                        offspring.add_node(ni_r, right_network.node(right_node_index).node_type().clone());
                    }
                }

                Alignment::Excess((&ni_l, &left_node_index), LeftOrRight::Left) => {
                    if c.prob_excess_left.flip(rng) {
                        offspring.add_node(ni_l, left_network.node(left_node_index).node_type().clone());
                    }
                }

                Alignment::Excess((&ni_r, &right_node_index), LeftOrRight::Right) => {
                    if c.prob_excess_right.flip(rng) {
                        offspring.add_node(ni_r, right_network.node(right_node_index).node_type().clone());
                    }
                }
            }
        });
    }

}

/// This is used to weight a link AlignmentMetric.
pub struct GenomeDistance {
    pub excess: f64,
    pub disjoint: f64,
    pub weight: f64,
}

impl<NT: NodeType> Distance<Genome<NT>> for GenomeDistance {
    fn distance(&self, genome_left: &Genome<NT>, genome_right: &Genome<NT>) -> f64 {
        let m = Genome::combined_alignment_metric(genome_left, genome_right).link_metric;

        if m.max_len == 0 {
            return 0.0;
        }

        self.excess * (m.excess as f64) / (m.max_len as f64) +
        self.disjoint * (m.disjoint as f64) / (m.max_len as f64) +
        self.weight *
        if m.matching > 0 {
            m.weight_distance / (m.matching as f64)
        } else {
            0.0
        }
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

        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(0, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(0, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(5), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(1, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(10), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(2, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(2, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(6), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(2, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(2, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(5), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(2, m.max_len);
        assert_eq!(1, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(6), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(3, m.max_len);
        assert_eq!(2, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(11), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(3, m.max_len);
        assert_eq!(2, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);
    }

    #[test]
    fn test_combined_align_metric() {
        let mut left = Genome::<NT>::new();
        let mut right = Genome::<NT>::new();

        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        left.add_node(NodeInnovation(5), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        left.add_node(NodeInnovation(10), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        right.add_node(NodeInnovation(6), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        right.add_node(NodeInnovation(5), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        left.add_node(NodeInnovation(6), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        right.add_node(NodeInnovation(11), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);
    }

}
