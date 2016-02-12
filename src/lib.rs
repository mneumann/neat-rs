#![feature(iter_arith, zero_one)]

extern crate rand;
extern crate rayon;
extern crate fixedbitset;

use rand::{Rng, Closed01};

pub mod traits;
pub mod innovation;
mod selection;
mod alignment;
pub mod crossover;
pub mod fitness;
pub mod population;
pub mod network;

pub fn is_probable<R: Rng>(prob: &Closed01<f32>, rng: &mut R) -> bool {
    if prob.0 < 1.0 {
        let v: f32 = rng.gen(); // half open [0, 1)
        debug_assert!(v >= 0.0 && v < 1.0);
        v < prob.0
    } else {
        true
    }
}
