use crate::operation::{Operation, Variable};

use super::OptimizerRunner;

pub struct SGDOptimizerRunner {
    lr: f32,
}

impl SGDOptimizerRunner {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl OptimizerRunner for SGDOptimizerRunner {
    fn step(&mut self, variables: Vec<&mut Variable>) {
        for var in variables {
            for y in 0..var.value.height() {
                for x in 0..var.value.width() {
                    let current_value = var.value[y][x];
                    let grad_value = var.grad[y][x];

                    let new_value = current_value - self.lr * grad_value;
                    var.value[y][x] = new_value;
                }
            }
        }
    }
}
