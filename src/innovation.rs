use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct Innovation(usize);

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct InnovationRange(Innovation, Innovation);

impl Innovation {
    pub fn new(n: usize) -> Innovation {
        Innovation(n)
    }

    pub fn is_within(&self, range: &InnovationRange) -> bool {
        self >= &range.0 && self <= &range.1
    }
}

// Returns the min and max innovation numbers (inclusive)
pub fn innovation_range<T>(map: &BTreeMap<Innovation, T>) -> InnovationRange {
    innovation_range_iter(map.keys().cloned()).unwrap()
}

// This requires that the iteration is in sorted order.
fn innovation_range_iter<I>(mut iter: I) -> Option<InnovationRange>
    where I: Iterator<Item = Innovation> + DoubleEndedIterator<Item = Innovation>
{
    iter.next().map(|min| {
        match iter.next_back() {
            Some(max) => InnovationRange(min, max),
            None => InnovationRange(min, min),
        }
    })
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
    let mut genome = BTreeMap::<Innovation, ()>::new();
    genome.insert(Innovation(50), ());
    assert_eq!(InnovationRange(Innovation(50), Innovation(50)),
               innovation_range(&genome));
    genome.insert(Innovation(20), ());
    assert_eq!(InnovationRange(Innovation(20), Innovation(50)),
               innovation_range(&genome));
    genome.insert(Innovation(100), ());
    assert_eq!(InnovationRange(Innovation(20), Innovation(100)),
               innovation_range(&genome));
    genome.insert(Innovation(0), ());
    assert_eq!(InnovationRange(Innovation(0), Innovation(100)),
               innovation_range(&genome));
}
