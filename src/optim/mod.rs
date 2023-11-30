pub mod sgd;

use crate::{
    model::Model,
    node::{variable::NodeVariable, Node},
};

pub trait OptimFunc {
    fn step(&mut self, variable: &mut NodeVariable);
}

pub struct Optimizer<TOptimFunc: OptimFunc> {
    variables: Vec<Node>,
    func: TOptimFunc,
}

impl<TOptimFunc: OptimFunc> Optimizer<TOptimFunc> {
    pub fn new(func: TOptimFunc) -> Self {
        Self {
            variables: Vec::new(),
            func,
        }
    }

    pub fn add_variables(&mut self, nodes: &Vec<Node>) {
        for node in nodes {
            if node.is_variable() {
                self.variables.push(node.clone());
            }
        }
    }

    pub fn add_model<TModelType: Model>(&mut self, model: &TModelType) {
        model.for_each_variable(&mut |var| {
            if var.is_variable() {
                self.variables.push(var.clone());
            }
        });
    }

    pub fn step(&mut self) {
        for v in &self.variables {
            let mut internal = v.get_internal();
            if let Some(variable) = internal.try_get_variable() {
                self.func.step(variable)
            }
        }
    }
}
