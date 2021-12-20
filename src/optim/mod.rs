use std::{cell::RefMut, ops::DerefMut};

use crate::matrix::{Matrix, MatrixRef};

pub mod sgd;

pub trait Optimizer {
    fn add_variable(&mut self, value: MatrixRef, grad: MatrixRef);

    fn step(&mut self);
}

pub trait OptimizerRunner {
    fn run(&mut self, variables: Vec<(&mut Matrix, &mut Matrix)>);
}

pub struct RunningOptimizer<O: OptimizerRunner + 'static> {
    variables: Vec<(MatrixRef, MatrixRef)>,
    runner: O,
}

impl<O: OptimizerRunner + 'static> RunningOptimizer<O> {
    pub fn new(runner: O) -> Self {
        Self {
            variables: Vec::new(),
            runner,
        }
    }
}

impl<O: OptimizerRunner + 'static> Optimizer for RunningOptimizer<O> {
    fn add_variable(&mut self, value: MatrixRef, grad: MatrixRef) {
        self.variables.push((value, grad));
    }

    fn step(&mut self) {
        let mut borrowed_variables = self
            .variables
            .iter_mut()
            .map(|(val, grad)| (val.get_mut(), grad.get_mut()))
            .collect::<Vec<(RefMut<Matrix>, RefMut<Matrix>)>>();

        let deref_variables = borrowed_variables
            .iter_mut()
            .map(|(val, grad)| (val.deref_mut(), grad.deref_mut()))
            .collect::<Vec<(&mut Matrix, &mut Matrix)>>();

        self.runner.run(deref_variables);
    }
}
