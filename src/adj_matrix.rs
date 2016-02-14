use fixedbitset::FixedBitSet;

#[derive(Debug, Clone)]
pub struct AdjMatrix {
    n: usize,
    m: FixedBitSet,
}

impl AdjMatrix {
    pub fn new(n: usize) -> AdjMatrix {
        AdjMatrix {
            n: n,
            m: FixedBitSet::with_capacity(n * n),
        }
    }

    pub fn set(&mut self, i: usize, j: usize) {
        self.m.insert(i * self.n + j);
    }

    fn contains(&self, i: usize, j: usize) -> bool {
        self.m.contains(i * self.n + j)
    }

    /// This is O(n**4) in worst case.
    /// XXX: This needs a test for correctness
    pub fn transitive_closure(mut self) -> AdjMatrix {
        loop {
            let mut counts = 0;
            for i in 0..self.n {
                for j in 0..self.n {
                    if self.contains(i, j) {
                        // look column for j
                        for k in 0..self.n {
                            if self.contains(j, k) {
                                if !self.contains(i, k) {
                                    self.set(i, k);
                                    counts += 1;
                                }
                            }
                        }
                    }
                }
            }
            if counts == 0 {
                break;
            }
        }
        self
    }

    pub fn unconnected_pairs_no_cycle(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.n {
            for j in 0..self.n {
                if i != j {
                    if !(self.contains(i, j) || self.contains(j, i)) {
                        // make sure we don't introduce a cycle
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs
    }
}
