use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct TimesRunner {
    value: f32,
}

impl UnaryOperationRunner for TimesRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|data_iter| data_iter.map(|mat_val| self.value * mat_val).collect()),
        )
    }

    fn grad(&self, child: &mut Operation, grad: &Matrix) {
        child.back_grad(Matrix::new(
            grad.height(),
            grad.width(),
            grad.chain_data(|di| di.map(|v| self.value * v).collect()),
        ));
    }
}

impl Operation {
    pub fn times(self, value: f32) -> Self {
        UnaryOperation::new(self, TimesRunner { value })
    }
}
