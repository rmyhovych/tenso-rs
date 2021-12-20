use std::{cell::Ref, slice::Iter};

use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct MeanRunner;

impl UnaryOperationRunner for MeanRunner {
    fn run(&self, input: Ref<Matrix>) -> Matrix {
        Matrix::from_const(
            1,
            1,
            input.chain_data(|data_iter: Iter<f32>| data_iter.sum::<f32>())
                / (input.width() * input.height()) as f32,
        )
    }

    fn grad(&self, input: Ref<Matrix>, _: Ref<Matrix>, delta: Matrix) -> Matrix {
        debug_assert_eq!(delta.width(), 1);
        debug_assert_eq!(delta.height(), 1);

        let mat_size = (input.width() * input.height()) as f32;

        let delta_val = delta[0][0];

        Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|child_data| child_data.map(|_| delta_val / mat_size).collect()),
        )
    }
}

impl Operation {
    pub fn mean(self) -> Self {
        UnaryOperation::new(self, MeanRunner)
    }
}
