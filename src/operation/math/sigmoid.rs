use std::cell::Ref;

use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct SigmoidRunner;

impl SigmoidRunner {
    fn sigmoid(val: f32) -> f32 {
        1.0 / (1.0 + (-val).exp())
    }

    fn sigmoid_prime(val: f32) -> f32 {
        let sig = Self::sigmoid(val);
        sig * (1.0 - sig)
    }
}

impl UnaryOperationRunner for SigmoidRunner {
    fn run(&self, input: Ref<Matrix>) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|data_iter| data_iter.map(|v| Self::sigmoid(*v)).collect()),
        )
    }

    fn grad(&self, input: Ref<Matrix>, _: Ref<Matrix>, delta: Matrix) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_zip_data(&delta, |child_data| {
                child_data
                    .map(|(ci, gr)| gr * Self::sigmoid_prime(*ci))
                    .collect()
            }),
        )
    }
}

impl Operation {
    pub fn sigmoid(self) -> Self {
        UnaryOperation::new(self, SigmoidRunner)
    }
}
