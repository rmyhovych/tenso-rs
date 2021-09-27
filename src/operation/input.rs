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

    pub fn with_value(value: Matrix) -> Operation {
        let mut placeholder = Self::new();
        placeholder.set_input(value);
        placeholder
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
        let value = Rc::new(RefCell::new(value));
        let grad = Rc::new(RefCell::new(Matrix::zeros(0, 0)));

        Operation::new(Self { value, grad })
    }
}

impl OperationBase for Variable {
    fn run(&mut self) -> Matrix {
        self.get_output()
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(
            self.value.borrow().height(),
            self.value.borrow().width(),
            1.0,
        );
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        let mut grad_borrow = self.grad.borrow_mut();
        let new_grad = if grad.width() == grad_borrow.width()
            && grad.height() == grad_borrow.height()
        {
            Matrix::new(
                grad.height(),
                grad.width(),
                grad.chain_zip_data(grad_borrow.deref(), |data_zip| {
                    data_zip.map(|(v0, v1)| v0 + v1).collect()
                }),
            )
        } else {
            grad
        };

        grad_borrow.set(new_grad);
    }

    fn get_output(&self) -> Matrix {
        self.value.borrow().clone()
    }

    fn set_input(&mut self, input: Matrix) {
        self.value.borrow_mut().set(input);
    }

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer) {
        optim.add_variable(Rc::clone(&self.value), Rc::clone(&self.grad));
    }
}

impl Matrix {
    pub fn as_variable(self) -> Operation {
        Variable::new(self)
    }
}
