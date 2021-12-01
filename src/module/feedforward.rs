use crate::{activation::Activation, operation::OperationRef};

use super::{
    linear::{LinearModuleData, LinearModuleRunner},
    ModuleBase, ModuleRunner, VariableGroup,
};

pub type FeedforwardModule = ModuleBase<FeedforwardLayers, FeedforwardModuleRunner>;

pub struct FeedforwardLayers {
    input_size: usize,

    layers: Vec<(usize, Box<dyn Activation>)>,
}

impl FeedforwardLayers {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            layers: Vec::new(),
        }
    }

    pub fn push(mut self, output_size: usize, activation: impl Activation + 'static) -> Self {
        self.layers.push((output_size, Box::new(activation)));
        self
    }
}

pub struct FeedforwardModuleRunner {
    layers: Vec<LinearModuleRunner>,
}

impl ModuleRunner<FeedforwardLayers> for FeedforwardModuleRunner {
    fn new(data: FeedforwardLayers, variables: &mut VariableGroup) -> Self {
        let mut layers: Vec<LinearModuleRunner> = Vec::new();

        let mut layer_input = data.input_size;
        for (layer_output, layer_activation) in data.layers {
            layers.push(LinearModuleRunner::new(
                LinearModuleData::from_boxed_activation(
                    layer_input,
                    layer_output,
                    layer_activation.into(),
                ),
                variables,
            ));

            layer_input = layer_output;
        }

        Self { layers }
    }

    fn run(&self, input: OperationRef) -> OperationRef {
        let mut op = input;
        for layer in &self.layers {
            op = layer.run(op);
        }

        op
    }
}
