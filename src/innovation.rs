use std::fmt::Debug;

/// Represents innovation numbers.
pub trait Innovation: Copy + Debug + Ord {
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnovationRange<T: Innovation> {
    Empty,
    Single(T),
    FromTo(T, T),
}

impl<T: Innovation> InnovationRange<T> {
    /// Creates an empty InnovationRange.

    pub fn empty() -> Self {
        InnovationRange::Empty
    }

    /// Returns `true` if the innovation range contains `innovation`. Otherwise `false`.

    pub fn contains(&self, innovation: &T) -> bool {
        match *self {
            InnovationRange::Empty => false,
            InnovationRange::Single(ref a) => a == innovation,
            InnovationRange::FromTo(ref min, ref max) => innovation >= min && innovation <= max,
        }
    }

    /// Extend the innovation range to include `innovation`.

    pub fn insert(&mut self, innovation: T) {
        let new = match *self {
            InnovationRange::Empty => InnovationRange::Single(innovation),
            InnovationRange::Single(a) => {
                if innovation == a {
                    InnovationRange::Single(a)
                } else if innovation < a {
                    InnovationRange::FromTo(innovation, a)
                } else {
                    InnovationRange::FromTo(a, innovation)
                }
            }
            InnovationRange::FromTo(min, max) => {
                if innovation < min {
                    InnovationRange::FromTo(innovation, max)
                } else if innovation > max {
                    InnovationRange::FromTo(min, innovation)
                } else {
                    debug_assert!(innovation >= min && innovation <= max);
                    InnovationRange::FromTo(min, max)
                }
            }
        };
        *self = new;
    }

    pub fn map<F, G>(self, f: F) -> InnovationRange<G>
        where F: Fn(T) -> G,
              G: Innovation
    {
        match self {
            InnovationRange::Empty => InnovationRange::Empty,
            InnovationRange::Single(a) => InnovationRange::Single(f(a)),
            InnovationRange::FromTo(min, max) => InnovationRange::FromTo(f(min), f(max)),
        }
    }
}
