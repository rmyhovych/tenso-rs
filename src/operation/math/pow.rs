use std::cell::Ref;

use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct PowRunner {
    power: f32,
}

impl UnaryOperationRunner for PowRunner {
    fn run(&self, input: Ref<Matrix>) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|data_iter| data_iter.map(|v| v.powf(self.power)).collect()),
        )
    }

    fn grad(&self, input: Ref<Matrix>, _: Ref<Matrix>, delta: Matrix) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_zip_data(&delta, |child_data| {
                child_data
                    .map(|(ci, gr)| gr * self.power * ci.powf(self.power - 1.0))
                    .collect()
            }),
        )
    }
}

impl Operation {
    pub fn pow(self, power: f32) -> Self {
        UnaryOperation::new(self, PowRunner { power })
    }
}
