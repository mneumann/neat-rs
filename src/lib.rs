#![feature(iter_arith, zero_one)]

extern crate rand;
extern crate rayon;
extern crate fixedbitset;


pub mod traits;
pub mod innovation;
mod selection;
mod alignment;
pub mod crossover;
pub mod mutate;
pub mod fitness;
pub mod population;
pub mod network;
pub mod prob;
mod adj_matrix;
