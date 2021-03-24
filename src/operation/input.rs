use crate::{matrix::Matrix, optim::Optimizer};

use super::{Operation, OperationBase};

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
}

/*------------------------------------------------------------------------------------------------*/

struct Variable {
    value: Matrix,
    grad: Matrix,
}

impl Variable {
    fn new(value: Matrix, optim: &mut dyn Optimizer) -> Operation {
        let grad = Matrix::zeros(0, 0);
        optim.add_var(value.clone(), grad.clone());

        Operation::new(Self { value, grad })
    }
}

impl OperationBase for Variable {
    fn run(&mut self) -> Matrix {
        self.get_output()
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(self.value.get_height(), self.value.get_width(), 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        let new_grad = if grad.get_width() == self.grad.get_width()
            && grad.get_height() == self.grad.get_height()
        {
            Matrix::new(
                grad.get_height(),
                grad.get_width(),
                grad.chain_zip_data(&self.grad, |data_zip| {
                    data_zip.map(|(v0, v1)| v0 + v1).collect()
                }),
            )
        } else {
            grad
        };

        self.grad.set(&new_grad);
    }

    fn get_output(&self) -> Matrix {
        self.value.clone()
    }

    fn set_input(&mut self, input: Matrix) {
        self.value.set(&input);
    }
}

impl Matrix {
    pub fn var_op(&self, optim: &mut dyn Optimizer) -> Operation {
        Variable::new(self.clone(), optim)
    }
}
