pub mod sgd;

use crate::node::{variable::NodeVariable, Node};

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

    pub fn add_variables(&mut self, nodes: Vec<Node>) {
        self.variables
            .append(&mut nodes.into_iter().filter(|n| n.is_variable()).collect());
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
