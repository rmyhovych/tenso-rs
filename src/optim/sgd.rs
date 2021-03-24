use super::{OptimVariables, Optimizer};
use crate::matrix::Matrix;

pub struct SGDOptimizer {
    variables: OptimVariables,
    lr: f32,
}

impl SGDOptimizer {
    pub fn new(lr: f32) -> Self {
        Self {
            variables: OptimVariables::new(),
            lr,
        }
    }
}

impl Optimizer for SGDOptimizer {
    fn add_var(&mut self, value: Matrix, grad: Matrix) {
        self.variables.add(value, grad);
    }

    fn step(&mut self) {
        for (value, grad) in &mut self.variables.variables {
            for y in 0..value.get_height() {
                for x in 0..value.get_width() {
                    let current_value = value.get_value(y, x);
                    let grad_value = grad.get_value(y, x);

                    let new_value = current_value - self.lr * grad_value;
                    value.set_value(new_value, y, x);
                }
            }

            grad.clear();
        }
    }
}
