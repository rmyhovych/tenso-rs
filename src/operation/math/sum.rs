use crate::{
    matrix::{Matrix, MatrixData},
    operation::{UnaryOperation, UnaryOperationRunner},
};

struct SumRunner;

impl UnaryOperationRunner for SumRunner {
    fn run(&self, input: &MatrixData) -> MatrixData {
        MatrixData::from_const(1, 1, input.chain_data(|data_iter| data_iter.sum::<f32>()))
    }

    fn grad(&self, input: &MatrixData, delta: &MatrixData) -> MatrixData {
        let delta_val = delta[0][0];

        MatrixData::new(
            input.height(),
            input.width(),
            input.chain_data(|child_data| child_data.map(|_| delta_val).collect()),
        )
    }
}

impl Matrix {
    pub fn sum(self) -> Self {
        UnaryOperation::new(self, SumRunner)
    }
}
