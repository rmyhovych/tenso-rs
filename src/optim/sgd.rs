use super::OptimizerRunner;
use crate::matrix::Matrix;

pub struct SGDOptimizerRunner {
    lr: f32,
}

impl SGDOptimizerRunner {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl OptimizerRunner for SGDOptimizerRunner {
    fn run(&mut self, variables: Vec<(&mut Matrix, &mut Matrix)>) {
        for (value, grad) in variables {
            for y in 0..value.height() {
                for x in 0..value.width() {
                    let current_value = value[y][x];
                    let grad_value = grad[y][x];

                    let new_value = current_value - self.lr * grad_value;
                    value[y][x] = new_value;
                }
            }

            grad.clear();
        }
    }
}
