#![feature(iter_arith, zero_one)]

extern crate rand;
extern crate rayon;

pub mod traits;
pub mod innovation;
mod selection;
mod alignment;
mod crossover;
pub mod fitness;
pub mod population;
pub mod network;
