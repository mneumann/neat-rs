use gene::Gene;
use sorted_vec::SortedUniqueVec;
use alignment_metric::AlignmentMetric;
use alignment::{Alignment, align};
use std::cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use innovation::Innovation;

#[derive(Debug, Clone)]
/// Wraps a  gene, providing the order by innovation number.
pub struct GeneWrapper<T: Gene> {
    gene: T,
}

impl<T: Gene> GeneWrapper<T> {
    pub fn as_ref(&self) -> &T {
        &self.gene
    }

    pub fn as_mut_ref(&mut self) -> &mut T {
        &mut self.gene
    }

    pub fn unwrap(self) -> T {
        self.gene
    }
}

impl<T: Gene> Gene for GeneWrapper<T> {
    fn innovation(&self) -> Innovation {
        self.gene.innovation()
    }

    fn weight_distance(&self, other: &Self) -> f64 {
        self.gene.weight_distance(&other.gene)
    }
}

impl<T: Gene> PartialEq for GeneWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&self.gene.innovation(), &other.gene.innovation())
    }
}

impl<T: Gene> Eq for GeneWrapper<T> {}

impl<T: Gene> PartialOrd for GeneWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.gene.innovation(), &other.gene.innovation())
    }
}

impl<T: Gene> Ord for GeneWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.gene.innovation(), &other.gene.innovation())
    }
}

#[derive(Debug, Clone)]
pub struct GeneList<T: Gene> {
    genes: SortedUniqueVec<GeneWrapper<T>>,
}

impl<T: Gene> GeneList<T> {
    pub fn new() -> Self {
        GeneList { genes: SortedUniqueVec::new() }
    }

    pub fn genes(&self) -> &SortedUniqueVec<GeneWrapper<T>> {
        &self.genes
    }

    pub fn insert(&mut self, gene: T) {
        self.genes.insert(GeneWrapper { gene: gene });
    }

    pub fn push(&mut self, gene: T) {
        self.genes.push(GeneWrapper { gene: gene });
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn find_by_innovation(&self, innovation: Innovation) -> Option<&T> {
        self.genes.find_by(|genew| genew.innovation().cmp(&innovation)).map(|genew| genew.as_ref())
    }

    pub fn find_by_innovation_mut(&mut self, innovation: Innovation) -> Option<&mut T> {
        match self.genes.index_by(|genew| genew.innovation().cmp(&innovation)) {
            Some(idx) => self.genes.get_mut(idx).map(|genew| genew.as_mut_ref()),
            None => None,
        }
    }

    pub fn insert_from_either_or_by_innovation(&mut self,
                                               innovation: Innovation,
                                               left: &Self,
                                               right: &Self) {
        if let None = self.find_by_innovation(innovation) {
            // insert gene from either left or right.
            if let Some(gene) = left.find_by_innovation(innovation) {
                self.insert(gene.clone());
            } else if let Some(gene) = right.find_by_innovation(innovation) {
                self.insert(gene.clone());
            } else {
                panic!();
            }
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&T) -> bool
    {
        self.genes.retain(|genew| f(genew.as_ref()));
    }

    pub fn alignment_metric(&self, right: &Self) -> AlignmentMetric {
        let mut m = AlignmentMetric::new();
        align(&self.genes,
              &right.genes,
              &mut |alignment| {
                  match alignment {
                      Alignment::Match(gene_left, gene_right) => {
                          m.matching += 1;
                          m.weight_distance += gene_left.weight_distance(gene_right).abs();
                      }
                      Alignment::DisjointLeft(_) | Alignment::DisjointRight(_) => {
                          m.disjoint += 1;
                      }
                      Alignment::ExcessLeftHead(_) |
                      Alignment::ExcessLeftTail(_) |
                      Alignment::ExcessRightHead(_) |
                      Alignment::ExcessRightTail(_) => {
                          m.excess += 1;
                      }
                  }
              });

        assert!(2 * m.matching + m.disjoint + m.excess == self.len() + right.len());
        return m;
    }
}
