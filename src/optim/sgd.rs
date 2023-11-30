use std::ops::{Mul, Sub};

use crate::node::variable::NodeVariable;

use super::OptimFunc;

pub struct OptimFuncSGD {
    lr: f32,
}

impl OptimFuncSGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl OptimFunc for OptimFuncSGD {
    fn step(&mut self, variable: &mut NodeVariable) {
        variable.access(&mut |value, gradient| {
            *value = value.sub(&gradient.mul(self.lr));
            gradient.take_clear();
        });
    }
}
