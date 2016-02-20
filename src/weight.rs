use rand::{Rng, Closed01};

/// Represents a connection weight. 
pub struct Weight(pub f64);

/// Represents the range within a connection weight can be.
pub enum WeightRange {
    /// Bipolar weight range is within [-magnitude, +magnitude].
    Bipolar {magnitude: f64},
    /// Unipolar weight range is within [0, +magnitude].
    Unipolar {magnitude: f64}
}

impl WeightRange {
    pub fn random_weight<R: Rng>(&self, rng: &mut R) -> Weight {
       let w: Closed01<f64> = rng.gen(); // f64 in the range [0, 1]
       match *self {
           WeightRange::Bipolar {magnitude} => {
               let w = Weight((2.0 * magnitude * w.0) - magnitude);
               debug_assert!(w.0 >= -(magnitude.abs()) && w.0 <= magnitude.abs());
               w
           }
           WeightRange::Unipolar {magnitude} => {
               let w = Weight(magnitude * w.0);
               if magnitude > 0.0 {
                   debug_assert!(w.0 >= 0.0 && w.0 <= magnitude);
               } else {
                   debug_assert!(w.0 >= magnitude && w.0 <= 0.0);
               }
               w
           }
       }
    }

    pub fn clip_weight(&self, weight: Weight) -> Weight {
        match *self {
            WeightRange::Bipolar {magnitude} => {
            }
            WeightRange::Unipolar {magnitude} => {
            }
        }
    }
}

/// Defines a perturbance method. 
pub enum WeightPerturbanceMethod {
    JiggleUniform  {magnitude: f64},
    JiggleGaussian {sigma: f64},
    Random,
}

impl WeightPerturbanceMethod {
    pub fn perturb<R: Rng>(&self, weight: Weight, range: &WeightRange, rng: &mut R) -> Weight {
        match *self {
            WeightPerturbanceMethod::Random => {
                range.random_weight(rng)
            }
            WeightPerturbanceMethod::JiggleUniform { magnitude } => {
                unimplemented!()
            }
            WeightPerturbanceMethod::JiggleGaussian {sigma} => {
                unimplemented!()
            }
        }
    }
}
