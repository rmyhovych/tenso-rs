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

    pub fn access<TAccessorType: FnMut(&mut Matrix, &mut Matrix)>(
        &mut self,
        accessor: &mut TAccessorType,
    ) {
        accessor(&mut self.value, &mut self.gradient);
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
