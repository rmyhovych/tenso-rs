use crate::{activation::Activation, matrix::Matrix, operation::OperationRef};

use super::{ModuleRunner, VariableGroup};

pub struct LinearModuleData {
    input_size: usize,
    output_size: usize,

    activation: Box<dyn Activation>,
}

impl LinearModuleData {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: impl Activation + 'static,
    ) -> Self {
        Self::from_boxed_activation(input_size, output_size, Box::new(activation))
    }

    pub fn from_boxed_activation(
        input_size: usize,
        output_size: usize,
        activation: Box<dyn Activation>,
    ) -> Self {
        Self {
            input_size,
            output_size,
            activation,
        }
    }
}

pub struct LinearModuleRunner {
    weights: OperationRef,
    biases: OperationRef,

    activation: Box<dyn Activation>,
}

impl ModuleRunner<LinearModuleData> for LinearModuleRunner {
    fn new(data: LinearModuleData, variables: &mut VariableGroup) -> Self {
        let weights = variables.add(Matrix::randn(data.output_size, data.input_size, 0.0, 1.0));
        let biases = variables.add(Matrix::randn(data.output_size, 1, 0.0, 1.0));

        Self {
            weights,
            biases,
            activation: data.activation,
        }
    }

    fn run(&self, input: OperationRef) -> OperationRef {
        self.activation
            .run(self.weights.clone().mmul(input) + self.biases.clone())
    }
}
