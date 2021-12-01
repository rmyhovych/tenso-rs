use std::slice::Iter;

use crate::{
    matrix::Matrix,
    operation::{UnaryOperation, UnaryOperationRunner},
};

use super::OperationRef;

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

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        debug_assert_eq!(delta.width(), 1);
        debug_assert_eq!(delta.height(), 1);

        let delta_val = delta[0][0];

        let mat_size = (input.width() * input.height()) as f32;
        let input_delta = Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|child_data| child_data.map(|_| delta_val / mat_size).collect()),
        );

        input_delta
    }
}

impl OperationRef {
    pub fn mean(self) -> Self {
        UnaryOperation::new(&self, MeanRunner)
    }
}
