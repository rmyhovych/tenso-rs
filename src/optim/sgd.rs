use std::ops::{Mul, Sub};

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
    fn step(&mut self, variable: &mut crate::node::variable::NodeVariable) {
        let gradient = variable.take_gradient().mul(self.lr);

        let value: &mut crate::matrix::Matrix = variable.get_value_mut();
        *value = value.sub(&gradient);
    }
}
