/// New type representing an innovation number.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct Innovation(usize);

impl Innovation {
    pub fn new(n: usize) -> Innovation {
        Innovation(n)
    }

    pub fn get(&self) -> usize {
        self.0
    }
}

impl Innovation {
    pub fn next(&mut self) -> Option<Innovation> {
        match self.0.checked_add(1) {
            Some(n) => {
                self.0 = n;
                Some(Innovation(n))
            }
            None => None,
        }
    }
}

#[test]
fn test_innovation_next() {
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
