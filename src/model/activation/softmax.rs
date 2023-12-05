use std::ops::Mul;

use crate::{
    matrix::Matrix,
    node::{op::unary::OpUnary, Node},
};

use super::Activation;

struct OpUnarySoftmax;
impl OpUnary for OpUnarySoftmax {
    fn run(&self, input: &Matrix) -> Matrix {
        let divisor = {
            let input_sum = input.sum()[[0, 0]];
            input_sum
        };

        input.mul(1.0 / divisor)
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {

        // 23,   -5
        // 0.1,  0.2
        // 0.33, 0.67
    }
}

impl Node {
    pub fn softmax(&self) -> Self {
        self.op_unary(OpUnarySoftmax)
    }
}

pub struct ActivationSoftmax;
impl Activation for ActivationSoftmax {
    fn run(&self, node: &Node) -> Node {
        node.softmax()
    }
}
