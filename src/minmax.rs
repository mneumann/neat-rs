pub struct MinMax<T: Ord + Clone> {
    min_max: Option<(T, T)>,
}

impl<T: Ord + Clone> MinMax<T> {
    pub fn new() -> Self {
        MinMax {
            min_max: None
        }
    }

    pub fn add_value(&mut self, value: &T) {
        if let None = self.min_max {
            self.min_max = Some((value.clone(), value.clone())); 
        }
        else if let Some((ref mut min, ref mut max)) = self.min_max {
            if value < min {
                *min = value.clone();
            }
            if value > max {
                *max = value.clone();
            }
        }
    }

    pub fn min_max(self) -> Option<(T, T)> {
        self.min_max
    }
}
