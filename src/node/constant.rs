use crate::matrix::Matrix;

use super::{Node, NodeInternal};

pub struct NodeConstant {
    value: Matrix,
}

impl NodeConstant {
    pub fn new(value: Matrix) -> Node {
        Node::new(Self { value })
    }
}

impl NodeInternal for NodeConstant {
    fn get_value(&self) -> &Matrix {
        &self.value
    }

    fn back_delta(&mut self, _delta: Matrix) {}
}
