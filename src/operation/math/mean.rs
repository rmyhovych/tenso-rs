use std::slice::Iter;

use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct MeanRunner;

impl UnaryOperationRunner for MeanRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::from_const(
            1,
            1,
            input.chain_data(|data_iter: Iter<f32>| data_iter.sum::<f32>())
                / (input.width() * input.height()) as f32,
        )
    }

    fn grad(&self, child: &mut Operation, grad: &Matrix) {
        debug_assert_eq!(grad.width(), 1);
        debug_assert_eq!(grad.height(), 1);

        let child_in = child.get_output();
        let mat_size = (child_in.width() * child_in.height()) as f32;

        let grad_val = grad[0][0];

        let child_in = child.get_output();
        let child_grad = Matrix::new(
            child_in.height(),
            child_in.width(),
            child_in.chain_data(|child_data| child_data.map(|_| grad_val / mat_size).collect()),
        );

        child.back_grad(child_grad);
    }
}

impl Operation {
    pub fn mean(self) -> Self {
        UnaryOperation::new(self, MeanRunner)
    }
}
