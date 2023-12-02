use std::{fmt::Display, ops::Deref};

use crate::{
    matrix::Matrix,
    node::{variable::NodeVariable, Node},
};

use super::{
    activation::{Activation, EmptyActivation},
    Model,
};

pub struct ModelLinear {
    weights: Node,
    biases: Node,
    activation: Box<dyn Activation>,
}

impl ModelLinear {
    pub fn new(size_in: usize, size_out: usize) -> Self {
        Self::new_model(size_in, size_out, EmptyActivation)
    }

    pub fn new_activated<TActivationType: Activation + 'static>(
        size_in: usize,
        size_out: usize,
        activation: TActivationType,
    ) -> Self {
        Self::new_model(size_in, size_out, activation)
    }

    fn new_model<TActivationType: Activation + 'static>(
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
}

impl Model for ModelLinear {
    fn run(&self, x: &Node) -> Node {
        self.activation
            .run(&self.weights.matmul(x).add(&self.biases))
    }

    fn for_each_variable<TFuncType: FnMut(&Node)>(&self, func: &mut TFuncType) {
        func(&self.weights);
        func(&self.biases);
    }
}

impl Display for ModelLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Weights:\n")?;
        self.weights.fmt(f)?;
        f.write_str("Biases:\n")?;
        self.biases.fmt(f)?;

        Ok(())
    }
}
