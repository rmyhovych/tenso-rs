use std::ops::{Deref, Mul};

use crate::{
    matrix::Matrix,
    node::{Node, NodeInternal},
};

pub trait OpUnary: 'static {
    fn run(&self, input: &Matrix) -> Matrix;
    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix;
}

pub struct NodeOpUnary<TOpUnary: OpUnary> {
    input: Node,
    result: Matrix,
    func: Box<TOpUnary>,
}

impl<TOpUnary: OpUnary> NodeOpUnary<TOpUnary> {
    pub fn new(func: TOpUnary, input: &Node) -> Node {
        let result = func.run(input.get_value().deref());
        Node::new(Self {
            input: input.clone(),
            result,
            func: Box::new(func),
        })
    }
}

impl<TOpUnary: OpUnary> NodeInternal for NodeOpUnary<TOpUnary> {
    fn get_value(&self) -> &Matrix {
        &self.result
    }

    fn back_delta(&mut self, delta: Matrix) {
        let grad_delta = self.func.grad(self.input.get_value().deref(), &delta);
        self.input.back_delta(grad_delta);
    }
}

/* --------------------------------------------------------------------------------- */

struct OpUnaryTimes {
    value: f32,
}
impl OpUnary for OpUnaryTimes {
    fn run(&self, input: &Matrix) -> Matrix {
        self.value * input
    }

    fn grad(&self, _input: &Matrix, delta: &Matrix) -> Matrix {
        self.value * delta
    }
}

struct OpUnarySum;
impl OpUnary for OpUnarySum {
    fn run(&self, input: &Matrix) -> Matrix {
        input.sum()
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        let delta_value = delta[[0, 0]];
        Matrix::new_value(input.size(), delta_value)
    }
}

struct OpUnaryPow {
    pow: f32,
}
impl OpUnary for OpUnaryPow {
    fn run(&self, input: &Matrix) -> Matrix {
        input.unary_operation(|v| v.powf(self.pow))
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        let derivative_pow = self.pow - 1.0;
        input
            .unary_operation(|v| self.pow * v.powf(derivative_pow))
            .mul(delta)
    }
}

struct OpUnaryTranspose;
impl OpUnary for OpUnaryTranspose {
    fn run(&self, input: &Matrix) -> Matrix {
        input.transpose()
    }

    fn grad(&self, _input: &Matrix, delta: &Matrix) -> Matrix {
        delta.transpose()
    }
}

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

/* --------------------------------------------------------------------------------- */

impl Node {
    pub fn op_unary<TOpUnary: OpUnary>(&self, func: TOpUnary) -> Self {
        NodeOpUnary::new(func, self)
    }

    pub fn times(&self, value: f32) -> Self {
        self.op_unary(OpUnaryTimes { value })
    }

    pub fn sum(&self) -> Self {
        self.op_unary(OpUnarySum)
    }

    pub fn pow(&self, pow: f32) -> Self {
        self.op_unary(OpUnaryPow { pow })
    }

    pub fn transpose(&self) -> Self {
        self.op_unary(OpUnaryTranspose)
    }

    pub fn sigmoid(&self) -> Self {
        self.op_unary(OpUnarySigmoid)
    }
}
