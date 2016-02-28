#![feature(iter_arith, zero_one)]

extern crate rand;
extern crate rayon;
#[macro_use]
extern crate log;
extern crate acyclic_network;
extern crate closed01;

pub mod traits;
pub mod innovation;
mod selection;
pub mod alignment;
pub mod alignment_metric;
pub mod mutate;
pub mod fitness;
pub mod population;
pub mod prob;
pub mod crossover;
pub mod genomes;
pub mod weight;
pub mod distribute;
