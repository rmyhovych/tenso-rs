use std::ops::Deref;

use crate::{
    matrix::Matrix,
    node::{variable::NodeVariable, Node},
};

use super::Model;

pub struct ModelLinear {
    weights: Node,
    biases: Node,
    activation: Box<dyn Fn(&Node) -> Node>,
}

impl ModelLinear {
    pub fn new(size_in: usize, size_out: usize) -> Self {
        Self::new_model(size_in, size_out, Self::empty_activation)
    }

    pub fn new_activated<TActivationType: Fn(&Node) -> Node + 'static>(
        size_in: usize,
        size_out: usize,
        activation: TActivationType,
    ) -> Self {
        Self::new_model(size_in, size_out, activation)
    }

    fn new_model<TActivationType: Fn(&Node) -> Node + 'static>(
        size_in: usize,
        size_out: usize,
        activation: TActivationType,
    ) -> Self {
        Self {
            weights: NodeVariable::new(Matrix::new_randn([size_out, size_in], 0.0, 1.0)),
            biases: NodeVariable::new(Matrix::new_randn([size_out, 1], 0.0, 1.0)),
            activation: Box::new(activation),
        }
    }

    fn empty_activation(node: &Node) -> Node {
        node.clone()
    }
}

impl Model for ModelLinear {
    fn run(&self, x: &Node) -> Node {
        self.activation.deref()(&self.weights.matmul(x).add(&self.biases))
    }

    fn for_each_variable<TFuncType: FnMut(&Node)>(&self, func: &mut TFuncType) {
        func(&self.weights);
        func(&self.biases);
    }
}
