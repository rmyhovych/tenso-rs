use std::{cell::RefCell, ops::DerefMut, rc::Rc};

use crate::matrix::Matrix;

pub mod sgd;

pub trait Optimizer {
    fn add_variable(&mut self, value: Rc<RefCell<Matrix>>, grad: Rc<RefCell<Matrix>>);

    fn step(&mut self);
}

pub trait OptimizerRunner {
    fn run(&mut self, value: &mut Matrix, grad: &mut Matrix);
}

pub struct RunningOptimizer<O: OptimizerRunner + 'static> {
    variables: Vec<(Rc<RefCell<Matrix>>, Rc<RefCell<Matrix>>)>,
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
    fn add_variable(&mut self, value: Rc<RefCell<Matrix>>, grad: Rc<RefCell<Matrix>>) {
        self.variables.push((value, grad));
    }

    fn step(&mut self) {
        for (value_rc, grad_rc) in &mut self.variables {
            self.runner.run(
                value_rc.borrow_mut().deref_mut(),
                grad_rc.borrow_mut().deref_mut(),
            );
        }
    }
}
