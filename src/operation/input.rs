use crate::{matrix::Matrix, optim::Optimizer};

use super::{Operation, OperationBase};

struct Constant {
    value: Matrix,
}

impl Constant {
    fn new(value: Matrix) -> Operation {
        Operation::new(Self { value })
    }
}

impl OperationBase for Constant {
    fn run(&mut self) -> &Matrix {
        self.get_output()
    }

    fn back(&mut self) {}

    fn back_grad(&mut self, _: Matrix) {}

    fn get_output(&self) -> &Matrix {
        &self.value
    }
}

impl Matrix {
    pub fn const_op(&self) -> Operation {
        Constant::new(self.clone())
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
    fn run(&mut self) -> &Matrix {
        self.get_output()
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(self.value.get_height(), self.value.get_width(), 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.grad.set(&grad);
    }

    fn get_output(&self) -> &Matrix {
        &self.value
    }
}

impl Matrix {
    pub fn var_op(&self, optim: &mut dyn Optimizer) -> Operation {
        Variable::new(self.clone(), optim)
    }
}
