use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

use super::OperationRef;

struct ReluRunner;

impl UnaryOperationRunner for ReluRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|data_iter| {
                data_iter.map(|v| if *v > 0.0 { *v } else { 0.0 }).collect()
            }),
        )
    }

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_zip_data(delta, |child_data| {
                child_data
                    .map(|(ci, gr)| if *ci > 0.0 { *gr } else { 0.0 })
                    .collect()
            }),
        )
    }
}

impl OperationRef {
    pub fn relu(self) -> Self {
        UnaryOperation::new(&self, ReluRunner)
    }
}
