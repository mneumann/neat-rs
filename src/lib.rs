#![feature(iter_arith, zero_one)]

extern crate rand;
extern crate rayon;
extern crate fixedbitset;
#[macro_use]
extern crate log;
// extern crate cppn;
extern crate sorted_vec;
extern crate acyclic_network;

pub mod traits;
pub mod innovation;
mod selection;
pub mod alignment;
pub mod alignment_metric;
pub mod mutate;
pub mod fitness;
pub mod population;
pub mod prob;

pub mod gene;
pub mod gene_list;
pub mod crossover;

pub mod genomes;
