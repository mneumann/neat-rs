use sorted_vec::SortedUniqueVec;

#[derive(Debug, PartialEq, Eq)]
pub enum Alignment<'a, T: 'a> {
    Match(&'a T, &'a T),
    ExcessLeftHead(&'a T),
    ExcessRightHead(&'a T),
    ExcessLeftTail(&'a T),
    ExcessRightTail(&'a T),
    DisjointLeft(&'a T),
    DisjointRight(&'a T),
}

impl<'a, T: 'a> Alignment<'a, T> {
    pub fn is_match(&self) -> bool {
        match self {
            &Alignment::Match(..) => true,
            _ => false,
        }
    }

    pub fn is_disjoint_left(&self) -> bool {
        match self {
            &Alignment::DisjointLeft(..) => true,
            _ => false,
        }
    }

    pub fn is_disjoint_right(&self) -> bool {
        match self {
            &Alignment::DisjointRight(..) => true,
            _ => false,
        }
    }

    pub fn is_disjoint(&self) -> bool {
        match self {
            &Alignment::DisjointLeft(..) |
            &Alignment::DisjointRight(..) => true,
            _ => false,
        }
    }

    pub fn is_excess_left(&self) -> bool {
        match self {
            &Alignment::ExcessLeftHead(..) |
            &Alignment::ExcessLeftTail(..) => true,
            _ => false,
        }
    }

    pub fn is_excess_right(&self) -> bool {
        match self {
            &Alignment::ExcessRightHead(..) |
            &Alignment::ExcessRightTail(..) => true,
            _ => false,
        }
    }

    pub fn is_excess(&self) -> bool {
        match self {
            &Alignment::ExcessLeftHead(..) |
            &Alignment::ExcessLeftTail(..) |
            &Alignment::ExcessRightHead(..) |
            &Alignment::ExcessRightTail(..) => true,
            _ => false,
        }
    }
}

/// Align the items of two sorted unique vectors.
pub fn align<'a, F, T>(a: &'a SortedUniqueVec<T>, b: &'a SortedUniqueVec<T>, f: &mut F)
    where F: FnMut(Alignment<'a, T>),
          T: Ord + Clone
{
    let mut left_iter = a.as_ref().iter().peekable();
    let mut right_iter = b.as_ref().iter().peekable();
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
            (Some(l), Some(r)) => {
                if l < r {
                    take = Take::OneLeft;
                } else if r < l {
                    take = Take::OneRight;
                } else {
                    take = Take::Both;
                }
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
                    f(Alignment::ExcessLeftHead(value));
                } else {
                    f(Alignment::DisjointLeft(value));
                }

                left_count += 1;
            }
            Take::OneRight => {
                let value = right_iter.next().unwrap();

                if left_count == 0 {
                    f(Alignment::ExcessRightHead(value));
                } else {
                    f(Alignment::DisjointRight(value));
                }

                right_count += 1;
            }
            Take::Both => {
                // two equal values
                let left_value = left_iter.next().unwrap();
                let right_value = right_iter.next().unwrap();
                debug_assert!(left_value.eq(right_value));

                f(Alignment::Match(left_value, right_value));

                left_count += 1;
                right_count += 1;
            }
            Take::AllLeft => {
                // There are no items left on the right side, so all items are ExcessLeftTail.
                for item in left_iter {
                    f(Alignment::ExcessLeftTail(item));
                }
                break;
            }
            Take::AllRight => {
                // There are no items left on the right side, so all items are ExcessRightTail.
                for item in right_iter {
                    f(Alignment::ExcessRightTail(item));
                }
                break;
            }
        }
    }
}

#[cfg(test)]
fn align_as_vec<'a, T>(a: &'a SortedUniqueVec<T>,
                       b: &'a SortedUniqueVec<T>)
                       -> Vec<Alignment<'a, T>>
    where T: Ord + Clone
{
    let mut c = Vec::new();
    align(a, b, &mut |alignment| c.push(alignment));
    c
}

#[test]
fn test_align() {
    use sorted_vec::is_sorted_unique;

    let mut s1 = SortedUniqueVec::new();
    s1.insert(0);
    s1.insert(1);
    s1.insert(5);
    s1.insert(8);
    assert!(is_sorted_unique(&s1));

    let mut s2 = SortedUniqueVec::new();
    s2.insert(1);
    s2.insert(5);
    s2.insert(7);
    s2.insert(9);
    s2.insert(55);
    assert!(is_sorted_unique(&s2));

    let mut r = SortedUniqueVec::new();
    align(&s1,
          &s2,
          &mut |alignment| {
              match alignment {
                  Alignment::Match(a, _b) => r.push((*a).clone()),
                  Alignment::ExcessLeftHead(a) |
                  Alignment::ExcessRightHead(a) |
                  Alignment::ExcessLeftTail(a) |
                  Alignment::ExcessRightTail(a) |
                  Alignment::DisjointLeft(a) |
                  Alignment::DisjointRight(a) => r.push((*a).clone()),
              }
          });

    assert!(is_sorted_unique(&r));
    assert_eq!(&[0, 1, 5, 7, 8, 9, 55][..], r.as_ref());
}

#[test]
fn test_align_as_vec() {
    let mut left = SortedUniqueVec::new();
    let mut right = SortedUniqueVec::new();
    // 40, 46, 50
    left.insert(50);
    left.insert(46);
    left.insert(40);

    // 45, 50, 51, 52
    right.insert(50);
    right.insert(45);
    right.insert(51);
    right.insert(52);

    let c = align_as_vec(&left, &right);
    assert_eq!(6, c.len());
    assert_eq!(Alignment::ExcessLeftHead(&40), c[0]);
    assert_eq!(Alignment::DisjointRight(&45), c[1]);
    assert_eq!(Alignment::DisjointLeft(&46), c[2]);
    assert_eq!(Alignment::Match(&50, &50), c[3]);
    assert_eq!(Alignment::ExcessRightTail(&51), c[4]);
    assert_eq!(Alignment::ExcessRightTail(&52), c[5]);
}
