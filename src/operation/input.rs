use std::ops::Deref;

use crate::{
    matrix::{Matrix, MatrixRef},
    optim::Optimizer,
};

use super::{Operation, OperationBase};

/*------------------------------------------------------------------------------------------------*/

pub struct InputPlaceholder {
    value: MatrixRef,
}

impl InputPlaceholder {
    pub fn new() -> Operation {
        Operation::new(Self {
            value: MatrixRef::new(Matrix::zeros(0, 0)),
        })
    }

    pub fn with_value(value: Matrix) -> Operation {
        let mut placeholder = Self::new();
        placeholder.set_input(MatrixRef::new(value));
        placeholder
    }
}

impl OperationBase for InputPlaceholder {
    fn run(&mut self) -> MatrixRef {
        self.get_output()
    }

    fn back(&mut self) {}

    fn back_grad(&mut self, _: Matrix) {}

    fn get_output(&self) -> MatrixRef {
        self.value.clone()
    }

    fn set_input(&mut self, input: MatrixRef) {
        self.value = input;
    }

    fn add_to_optimizer(&self, _: &mut dyn Optimizer) {}
}

/*------------------------------------------------------------------------------------------------*/

struct Variable {
    value: MatrixRef,
    grad: MatrixRef,
}

impl Variable {
    fn new(value: Matrix) -> Operation {
        let value = MatrixRef::new(value);
        let grad = MatrixRef::new(Matrix::zeros(0, 0));

        Operation::new(Self { value, grad })
    }
}

impl OperationBase for Variable {
    fn run(&mut self) -> MatrixRef {
        self.get_output()
    }

    fn back_grad(&mut self, grad: Matrix) {
        let mut grad_borrow = self.grad.get_mut();
        let new_grad =
            if grad.width() == grad_borrow.width() && grad.height() == grad_borrow.height() {
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

    fn get_output(&self) -> MatrixRef {
        self.value.clone()
    }

    fn set_input(&mut self, input: MatrixRef) {
        self.value.set(input.get().clone());
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
