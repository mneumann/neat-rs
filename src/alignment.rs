use std::cmp::Ordering;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum LeftOrRight {
    Left,
    Right,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Alignment<T> {
    Match(T, T),
    Excess(T, LeftOrRight),
    Disjoint(T, LeftOrRight),
}

impl<T> Alignment<T> {
    pub fn is_match(&self) -> bool {
        match *self {
            Alignment::Match(..) => true,
            _ => false,
        }
    }

    pub fn get_left(&self) -> Option<&T> {
        match *self {
            Alignment::Excess(ref a, LeftOrRight::Left) => Some(a),
            Alignment::Disjoint(ref a, LeftOrRight::Left) => Some(a),
            _ => None,
        }
    }

    pub fn get_right(&self) -> Option<&T> {
        match *self {
            Alignment::Excess(ref a, LeftOrRight::Right) => Some(a),
            Alignment::Disjoint(ref a, LeftOrRight::Right) => Some(a),
            _ => None,
        }
    }

    pub fn is_left(&self) -> bool {
        self.get_left().is_some()
    }

    pub fn is_right(&self) -> bool {
        self.get_right().is_some()
    }

    pub fn is_disjoint_left(&self) -> bool {
        match *self {
            Alignment::Disjoint(_, LeftOrRight::Left) => true,
            _ => false,
        }
    }

    pub fn is_disjoint_right(&self) -> bool {
        match *self {
            Alignment::Disjoint(_, LeftOrRight::Right) => true,
            _ => false,
        }
    }

    pub fn is_disjoint(&self) -> bool {
        match *self {
            Alignment::Disjoint(..) => true,
            _ => false,
        }
    }

    pub fn is_excess_left(&self) -> bool {
        match *self {
            Alignment::Excess(_, LeftOrRight::Left) => true,
            _ => false,
        }
    }

    pub fn is_excess_right(&self) -> bool {
        match *self {
            Alignment::Excess(_, LeftOrRight::Right) => true,
            _ => false,
        }
    }

    pub fn is_excess(&self) -> bool {
        match *self {
            Alignment::Excess(..) => true,
            _ => false,
        }
    }
}

/// Align the items of two sorted (unique) iterators.
pub fn align_sorted_iterators<CMP, F, I>(a: I, b: I, cmp: CMP, mut f: F)
    where CMP: Fn(&I::Item, &I::Item) -> Ordering,
          F: FnMut(Alignment<I::Item>),
          I: Iterator
{
    let mut left_iter = a.peekable();
    let mut right_iter = b.peekable();
    let mut left_count = 0;
    let mut right_count = 0;

    enum Take {
        OneLeft,
        OneRight,
        Both,
        AllLeft,
        AllRight,
    };

    loop {
        let take;

        match (left_iter.peek(), right_iter.peek()) {
            (Some(ref l), Some(ref r)) => {
                take = match cmp(l, r) {
                    Ordering::Less => Take::OneLeft,
                    Ordering::Greater => Take::OneRight,
                    Ordering::Equal => Take::Both,
                };
            }
            (Some(_), None) => {
                take = Take::AllLeft;
            }
            (None, Some(_)) => {
                take = Take::AllRight;
            }
            (None, None) => {
                break;
            }
        }

        match take {
            Take::OneLeft => {
                let value = left_iter.next().unwrap();

                if right_count == 0 {
                    // left head
                    f(Alignment::Excess(value, LeftOrRight::Left));
                } else {
                    f(Alignment::Disjoint(value, LeftOrRight::Left));
                }

                left_count += 1;
            }
            Take::OneRight => {
                let value = right_iter.next().unwrap();

                if left_count == 0 {
                    // right head
                    f(Alignment::Excess(value, LeftOrRight::Right));
                } else {
                    f(Alignment::Disjoint(value, LeftOrRight::Right));
                }

                right_count += 1;
            }
            Take::Both => {
                // two equal values
                let left_value = left_iter.next().unwrap();
                let right_value = right_iter.next().unwrap();
                debug_assert!(cmp(&left_value, &right_value) == Ordering::Equal);

                f(Alignment::Match(left_value, right_value));

                left_count += 1;
                right_count += 1;
            }
            Take::AllLeft => {
                // There are no items left on the right side, so all items are ExcessLeftTail.
                for item in left_iter {
                    // left tail
                    f(Alignment::Excess(item, LeftOrRight::Left));
                }
                break;
            }
            Take::AllRight => {
                // There are no items left on the right side, so all items are ExcessRightTail.
                for item in right_iter {
                    f(Alignment::Excess(item, LeftOrRight::Right));
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use super::{Alignment, align_sorted_iterators, LeftOrRight};

    fn align_as_vec<I>(a: I, b: I) -> Vec<Alignment<I::Item>>
        where I::Item: Ord + Clone,
              I: Iterator
    {
        let mut c = Vec::new();
        align_sorted_iterators(a, b, Ord::cmp, |alignment| c.push(alignment));
        c
    }

    #[test]
    fn test_align_sorted_iterators() {
        let mut s1 = BTreeSet::<usize>::new();
        s1.insert(0);
        s1.insert(1);
        s1.insert(5);
        s1.insert(8);

        let mut s2 = BTreeSet::<usize>::new();
        s2.insert(1);
        s2.insert(5);
        s2.insert(7);
        s2.insert(9);
        s2.insert(55);

        let mut r = Vec::new();
        super::align_sorted_iterators(s1.iter().cloned(),
                                      s2.iter().cloned(),
                                      Ord::cmp,
                                      |alignment| {
                                          match alignment {
                                              Alignment::Match(a, _b) => r.push(a),
                                              Alignment::Excess(a, _) => r.push(a),
                                              Alignment::Disjoint(a, _) => r.push(a),
                                          }
                                      });

        assert_eq!(vec![0, 1, 5, 7, 8, 9, 55], r);
    }

    #[test]
    fn test_align_as_vec() {
        let mut left = BTreeSet::<usize>::new();
        let mut right = BTreeSet::<usize>::new();

        // 40, 46, 50
        left.insert(50);
        left.insert(46);
        left.insert(40);

        // 45, 50, 51, 52
        right.insert(50);
        right.insert(45);
        right.insert(51);
        right.insert(52);

        let c = align_as_vec(left.iter().cloned(), right.iter().cloned());
        assert_eq!(6, c.len());
        assert_eq!(Alignment::Excess(40, LeftOrRight::Left), c[0]);
        assert_eq!(Alignment::Disjoint(45, LeftOrRight::Right), c[1]);
        assert_eq!(Alignment::Disjoint(46, LeftOrRight::Left), c[2]);
        assert_eq!(Alignment::Match(50, 50), c[3]);
        assert_eq!(Alignment::Excess(51, LeftOrRight::Right), c[4]);
        assert_eq!(Alignment::Excess(52, LeftOrRight::Right), c[5]);
    }
}
