use std::collections::BTreeMap;
use super::alignment::Alignment;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct Innovation(usize);

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct InnovationRange(Innovation, Innovation);

#[derive(Debug, Clone)]
pub struct InnovationContainer<T> {
    pub map: BTreeMap<Innovation, T>,
}

impl Iterator for Innovation {
    type Item = Innovation;
    fn next(&mut self) -> Option<Innovation> {
        match self.0.checked_add(1) {
            Some(n) => {
                self.0 = n;
                Some(Innovation(n))
            }
            None => None,
        }
    }
}

impl Innovation {
    pub fn new(n: usize) -> Innovation {
        Innovation(n)
    }

    pub fn is_within(&self, range: &InnovationRange) -> bool {
        self >= &range.0 && self <= &range.1
    }

    pub fn get(&self) -> usize {
        self.0
    }
}

impl InnovationRange {
    // NOTE: This requires that the iteration is in sorted order.
    fn from_sorted_iter<I>(mut iter: I) -> Option<InnovationRange>
        where I: Iterator<Item = Innovation> + DoubleEndedIterator<Item = Innovation>
    {
        iter.next().map(|min| {
            match iter.next_back() {
                Some(max) => InnovationRange(min, max),
                None => InnovationRange(min, min),
            }
        })
    }
}

impl<T> InnovationContainer<T> {
    pub fn new() -> InnovationContainer<T> {
        InnovationContainer { map: BTreeMap::new() }
    }

    pub fn insert(&mut self, innov: Innovation, data: T) {
        // XXX
        assert!(!self.map.contains_key(&innov));
        self.map.insert(innov, data);
    }

    pub fn get(&self, innov: &Innovation) -> Option<&T> {
        self.map.get(innov)
    }

    pub fn get_mut(&mut self, innov: &Innovation) -> Option<&mut T> {
        self.map.get_mut(innov)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    // Returns the min and max innovation numbers (inclusive)
    pub fn innovation_range(&self) -> InnovationRange {
        InnovationRange::from_sorted_iter(self.map.keys().cloned()).unwrap()
    }

    fn align_as_container<'a>(&'a self,
                              right: &'a InnovationContainer<T>)
                              -> InnovationContainer<Alignment<'a, T>> {
        let mut c = InnovationContainer::new();
        self.align(right, &mut |innov, alignment| c.insert(innov, alignment));
        c
    }

    pub fn align<'a, F>(&'a self, right: &'a InnovationContainer<T>, f: &mut F)
        where F: FnMut(Innovation, Alignment<'a, T>)
    {
        let range_left = self.innovation_range();
        let range_right = right.innovation_range();

        for (innov_left, gene_left) in self.map.iter() {
            if innov_left.is_within(&range_right) {
                match right.get(innov_left) {
                    Some(gene_right) => f(*innov_left, Alignment::Match(gene_left, gene_right)),
                    None => f(*innov_left, Alignment::DisjointLeft(gene_left)),
                }
            } else {
                f(*innov_left, Alignment::ExcessLeft(gene_left))
            }
        }

        for (innov_right, gene_right) in right.map.iter() {
            if innov_right.is_within(&range_left) {
                if !self.map.contains_key(innov_right) {
                    f(*innov_right, Alignment::DisjointRight(gene_right))
                }
            } else {
                f(*innov_right, Alignment::ExcessRight(gene_right))
            }
        }
    }
}

#[test]
fn test_innovation() {
    let r = InnovationRange(Innovation(0), Innovation(100));
    assert_eq!(true, Innovation(0).is_within(&r));
    assert_eq!(true, Innovation(50).is_within(&r));
    assert_eq!(true, Innovation(100).is_within(&r));
    assert_eq!(false, Innovation(101).is_within(&r));
    let r = InnovationRange(Innovation(1), Innovation(100));
    assert_eq!(false, Innovation(0).is_within(&r));
    assert_eq!(true, Innovation(1).is_within(&r));
    assert_eq!(true, Innovation(50).is_within(&r));
    assert_eq!(true, Innovation(100).is_within(&r));
    assert_eq!(false, Innovation(101).is_within(&r));
}

#[test]
fn test_innovation_range() {
    let mut genome = InnovationContainer::new();
    genome.insert(Innovation(50), ());
    assert_eq!(InnovationRange(Innovation(50), Innovation(50)),
               genome.innovation_range());
    genome.insert(Innovation(20), ());
    assert_eq!(InnovationRange(Innovation(20), Innovation(50)),
               genome.innovation_range());
    genome.insert(Innovation(100), ());
    assert_eq!(InnovationRange(Innovation(20), Innovation(100)),
               genome.innovation_range());
    genome.insert(Innovation(0), ());
    assert_eq!(InnovationRange(Innovation(0), Innovation(100)),
               genome.innovation_range());
}

#[test]
fn test_innovation_alignment() {
    let mut left = InnovationContainer::new();
    let mut right = InnovationContainer::new();
    left.insert(Innovation(50), ());
    left.insert(Innovation(46), ());
    left.insert(Innovation(40), ());
    right.insert(Innovation(50), ());
    right.insert(Innovation(45), ());
    right.insert(Innovation(51), ());
    right.insert(Innovation(52), ());

    let c = left.align_as_container(&right);
    assert_eq!(6, c.len());
    assert_eq!(true, c.get(&Innovation(50)).unwrap().is_match());
    assert_eq!(true, c.get(&Innovation(40)).unwrap().is_excess_left());
    assert_eq!(true, c.get(&Innovation(45)).unwrap().is_disjoint_right());
    assert_eq!(true, c.get(&Innovation(46)).unwrap().is_disjoint_left());
    assert_eq!(true, c.get(&Innovation(51)).unwrap().is_excess_right());
    assert_eq!(true, c.get(&Innovation(52)).unwrap().is_excess_right());
}

#[test]
fn test_innovation_iter() {
    let mut innovations = Innovation(1);
    assert_eq!(Innovation(1), innovations);
    assert_eq!(Some(Innovation(2)), innovations.next());
    assert_eq!(Some(Innovation(3)), innovations.next());
    assert_eq!(Some(Innovation(4)), innovations.next());
    assert_eq!(Some(Innovation(5)), innovations.next());

    let mut innovations = Innovation(usize::max_value() - 1);
    assert_eq!(Some(Innovation(usize::max_value())), innovations.next());
    assert_eq!(None, innovations.next());
}
