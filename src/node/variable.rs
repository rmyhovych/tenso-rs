use crate::matrix::Matrix;

use super::{Node, NodeInternal};

pub struct NodeVariable {
    value: Matrix,
    gradient: Matrix,
}

impl NodeVariable {
    pub fn new(value: Matrix) -> Node {
        let gradient = Matrix::new_zero(value.size());
        Node::new(Self { value, gradient })
    }

    pub fn get_value_mut(&mut self) -> &mut Matrix {
        &mut self.value
    }

    pub fn take_gradient(&mut self) -> Matrix {
        self.gradient.take_clear()
    }
}

impl NodeInternal for NodeVariable {
    fn get_value(&self) -> &Matrix {
        &self.value
    }

    fn back_delta(&mut self, delta: Matrix) {
        self.gradient = &self.gradient + &delta;
    }

    fn try_get_variable(&mut self) -> Option<&mut NodeVariable> {
        Some(self)
    }
}
