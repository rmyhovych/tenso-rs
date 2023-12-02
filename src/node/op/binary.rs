use std::ops::Deref;

use crate::{
    matrix::Matrix,
    node::{Node, NodeInternal},
};

pub trait OpBinary: 'static {
    fn run(&self, input: (&Matrix, &Matrix)) -> Matrix;
    fn grad(&self, input: (&Matrix, &Matrix), delta: &Matrix) -> (Matrix, Matrix);
}

pub struct NodeOpBinary<TOpBinary: OpBinary> {
    input: (Node, Node),
    result: Matrix,
    func: Box<TOpBinary>,
}

impl<TOpBinary: OpBinary> NodeOpBinary<TOpBinary> {
    pub fn new(func: TOpBinary, input: (&Node, &Node)) -> Node {
        let result = func.run((input.0.get_value().deref(), input.1.get_value().deref()));
        Node::new(Self {
            input: (input.0.clone(), input.1.clone()),
            result,
            func: Box::new(func),
        })
    }
}

impl<TOpBinary: OpBinary> NodeInternal for NodeOpBinary<TOpBinary> {
    fn get_value(&self) -> &Matrix {
        &self.result
    }

    fn back_delta(&mut self, delta: Matrix) {
        let grad_delta = self.func.grad(
            (
                self.input.0.get_value().deref(),
                self.input.1.get_value().deref(),
            ),
            &delta,
        );
        self.input.0.back_delta(grad_delta.0);
        self.input.1.back_delta(grad_delta.1);
    }
}

/* --------------------------------------------------------------------------------- */

struct OpBinaryAdd;
impl OpBinary for OpBinaryAdd {
    fn run(&self, input: (&Matrix, &Matrix)) -> Matrix {
        input.0 + input.1
    }

    fn grad(&self, _input: (&Matrix, &Matrix), delta: &Matrix) -> (Matrix, Matrix) {
        (delta.clone(), delta.clone())
    }
}

struct OpBinaryMul;
impl OpBinary for OpBinaryMul {
    fn run(&self, input: (&Matrix, &Matrix)) -> Matrix {
        input.0 * input.1
    }

    fn grad(&self, input: (&Matrix, &Matrix), delta: &Matrix) -> (Matrix, Matrix) {
        (delta * input.1, delta * input.0)
    }
}

struct OpBinaryMatMul;
impl OpBinary for OpBinaryMatMul {
    fn run(&self, input: (&Matrix, &Matrix)) -> Matrix {
        input.0.matmul(input.1)
    }

    fn grad(&self, input: (&Matrix, &Matrix), delta: &Matrix) -> (Matrix, Matrix) {
        let delta_left = delta.matmul(&input.1.transpose());
        let delta_right = input.0.transpose().matmul(&delta);

        (delta_left, delta_right)
    }
}

/* --------------------------------------------------------------------------------- */

impl Node {
    pub fn op_binary<TOpBinary: OpBinary>(&self, func: TOpBinary, other: &Node) -> Self {
        NodeOpBinary::new(func, (self, other))
    }

    pub fn add(&self, other: &Node) -> Self {
        self.op_binary(OpBinaryAdd, &other)
    }

    pub fn sub(&self, other: &Node) -> Self {
        self.add(&other.times(-1.0))
    }

    pub fn mul(&self, other: &Node) -> Self {
        self.op_binary(OpBinaryMul, &other)
    }

    pub fn matmul(&self, other: &Node) -> Self {
        self.op_binary(OpBinaryMatMul, &other)
    }
}
