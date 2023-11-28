use crate::matrix::Matrix;

use super::NodeInternal;

pub struct NodeVariable {
    value: Matrix,
    gradient: Matrix,
}

impl NodeInternal for NodeVariable {
    fn get_value(&self) -> &Matrix {
        &self.value
    }

    fn get_value_mut(&mut self) -> &mut Matrix {
        &mut self.value
    }

    fn back_delta(&mut self, delta: Matrix) {
        self.gradient = &self.gradient + &delta;
    }

    fn is_variable(&mut self) -> Option<&mut NodeVariable> {
        Some(self)
    }
}
