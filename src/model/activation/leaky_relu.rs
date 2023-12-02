use std::ops::Mul;

use crate::{
    matrix::Matrix,
    node::{op::unary::OpUnary, Node},
};

use super::Activation;

pub struct OpUnaryLeakyRelu {
    neg_slope: f32,
}
impl OpUnary for OpUnaryLeakyRelu {
    fn run(&self, input: &Matrix) -> Matrix {
        input.unary_operation(|v| v * if v < 0.0 { self.neg_slope } else { 1.0 })
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        delta.mul(&input.unary_operation(|v| if v > 0.0 { 1.0 } else { self.neg_slope }))
    }
}

impl Node {
    pub fn leaky_relu(&self, neg_slope: f32) -> Self {
        debug_assert!(neg_slope >= 0.0 && neg_slope <= 1.0);
        self.op_unary(OpUnaryLeakyRelu { neg_slope })
    }
}

pub struct ActivationLeakyRelu {
    neg_slope: f32,
}

impl ActivationLeakyRelu {
    pub fn new(neg_slope: f32) -> Self {
        Self { neg_slope }
    }
}

impl Activation for ActivationLeakyRelu {
    fn run(&self, node: &Node) -> Node {
        node.leaky_relu(self.neg_slope)
    }
}
