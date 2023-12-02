use std::ops::Mul;

use crate::{
    matrix::Matrix,
    node::{op::unary::OpUnary, Node},
};

use super::Activation;

pub struct OpUnaryRelu;
impl OpUnary for OpUnaryRelu {
    fn run(&self, input: &Matrix) -> Matrix {
        input.unary_operation(|v| v.max(0.0))
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        delta.mul(&input.unary_operation(|v| if v > 0.0 { 1.0 } else { 0.0 }))
    }
}

impl Node {
    pub fn relu(&self) -> Self {
        self.op_unary(OpUnaryRelu)
    }
}

pub struct ActivationRelu;
impl Activation for ActivationRelu {
    fn run(&self, node: &Node) -> Node {
        node.relu()
    }
}
