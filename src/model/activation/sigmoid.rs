use std::ops::Mul;

use crate::{
    matrix::Matrix,
    node::{op::unary::OpUnary, Node},
};

use super::Activation;

struct OpUnarySigmoid;
impl OpUnarySigmoid {
    fn sigmoid(val: f32) -> f32 {
        1.0 / (1.0 + (-val).exp())
    }
}
impl OpUnary for OpUnarySigmoid {
    fn run(&self, input: &Matrix) -> Matrix {
        input.unary_operation(|v| Self::sigmoid(v))
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        input
            .unary_operation(|v| {
                let s = Self::sigmoid(v);
                s * (1.0 - s)
            })
            .mul(delta)
    }
}

impl Node {
    pub fn sigmoid(&self) -> Self {
        self.op_unary(OpUnarySigmoid)
    }
}

pub struct ActivationSigmoid;
impl Activation for ActivationSigmoid {
    fn run(&self, node: &Node) -> Node {
        node.sigmoid()
    }
}
