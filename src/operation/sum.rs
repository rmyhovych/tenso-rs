use crate::{
    matrix::Matrix,
    operation::{UnaryOperation, UnaryOperationRunner},
};

use super::OperationRef;

struct SumRunner;

impl UnaryOperationRunner for SumRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::from_const(1, 1, input.chain_data(|data_iter| data_iter.sum::<f32>()))
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        debug_assert_eq!(delta.width(), 1);
        debug_assert_eq!(delta.height(), 1);

        let grad_val = delta[0][0];

        Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|child_data| child_data.map(|_| grad_val).collect()),
        )
    }
}

impl OperationRef {
    pub fn sum(self) -> Self {
        UnaryOperation::new(&self, SumRunner)
    }
}
