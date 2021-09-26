use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{matrix::Matrix, optim::Optimizer};

use super::{Operation, OperationBase};

/*------------------------------------------------------------------------------------------------*/

pub struct InputPlaceholder {
    value: Matrix,
}

impl InputPlaceholder {
    pub fn new() -> Operation {
        Operation::new(Self {
            value: Matrix::zeros(0, 0),
        })
    }
}

impl OperationBase for InputPlaceholder {
    fn run(&mut self) -> Matrix {
        self.get_output()
    }

    fn back(&mut self) {}

    fn back_grad(&mut self, _: Matrix) {}

    fn get_output(&self) -> Matrix {
        self.value.clone()
    }

    fn set_input(&mut self, input: Matrix) {
        self.value = input;
    }

    fn add_to_optimizer(&self, _: &mut dyn Optimizer) {}
}

/*------------------------------------------------------------------------------------------------*/

struct Variable {
    value: Rc<RefCell<Matrix>>,
    grad: Rc<RefCell<Matrix>>,
}

impl Variable {
    fn new(value: Matrix) -> Operation {
        let grad = Matrix::zeros(0, 0);
        Operation::new(Self {
            value: Rc::new(RefCell::new(value)),
            grad: Rc::new(RefCell::new(grad)),
        })
    }
}

impl OperationBase for Variable {
    fn run(&mut self) -> Matrix {
        self.get_output()
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(
            self.value.borrow().get_height(),
            self.value.borrow().get_width(),
            1.0,
        );
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        let new_grad = if grad.get_width() == self.grad.borrow().get_width()
            && grad.get_height() == self.grad.borrow().get_height()
        {
            Matrix::new(
                grad.get_height(),
                grad.get_width(),
                grad.chain_zip_data(&self.grad.borrow().deref(), |data_zip| {
                    data_zip.map(|(v0, v1)| v0 + v1).collect()
                }),
            )
        } else {
            grad
        };

        *self.grad.borrow_mut() = new_grad;
    }

    fn get_output(&self) -> Matrix {
        self.value.borrow().clone()
    }

    fn set_input(&mut self, input: Matrix) {
        *self.value.borrow_mut() = input;
    }

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer) {
        optim.add_variable(self.value.clone(), self.grad.clone());
    }
}

impl Matrix {
    pub fn as_variable(self) -> Operation {
        Variable::new(self)
    }
}
