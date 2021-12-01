use std::cell::RefMut;

use crate::{
    module::Module,
    operation::{Operation, OperationRef, Variable},
};

pub mod sgd;

pub trait OptimizerRunner {
    fn step(&mut self, variables: Vec<&mut Variable>);
}

pub struct OptimizerBase<R: OptimizerRunner> {
    operations: Vec<OperationRef>,
    runner: R,
}

impl<R: OptimizerRunner> OptimizerBase<R> {
    pub fn new(runner: R, module: &dyn Module) -> Self {
        Self {
            operations: module.get_variables().clone(),
            runner,
        }
    }

    pub fn step(&mut self) {
        let mut op_borrowed: Vec<RefMut<dyn Operation>> =
            self.operations.iter_mut().map(|op| op.as_mut()).collect();

        self.runner.step(
            op_borrowed
                .iter_mut()
                .filter_map(|r| r.as_variable())
                .collect(),
        );
    }

    pub fn zero_grad(&mut self) {
        for op in &mut self.operations {
            let mut op_borrow = op.as_mut();
            if let Some(var) = op_borrow.as_variable() {
                var.zero_grad();
            }
        }
    }
}
