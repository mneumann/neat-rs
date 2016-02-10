#![feature(iter_arith, zero_one)]

extern crate rand;

use std::cmp;
use innovation::{Innovation, InnovationContainer};
use alignment::Alignment;
use rand::{Rng, Closed01};
use mate::Mate;
use traits::Distance;
use traits::Genotype;
use fitness::Fitness;

mod traits;
mod innovation;
mod selection;
mod alignment;
mod crossover;
mod mate;
mod fitness;
mod population;
mod network;
