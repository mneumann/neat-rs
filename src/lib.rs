#![feature(iter_arith, zero_one)]

extern crate rand;
extern crate rayon;

mod traits;
pub mod innovation;
mod selection;
mod alignment;
mod crossover;
mod mate;
mod fitness;
pub mod population;
pub mod network;
