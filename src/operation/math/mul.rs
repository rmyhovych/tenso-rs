use std::{cell::Ref, ops::Mul};

use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner, Operation},
};

struct MulRunner;

impl BinaryOperationRunner for MulRunner {
    fn run(&self, input_left: Ref<Matrix>, input_right: Ref<Matrix>) -> Matrix {
        debug_assert_eq!(input_left.width(), input_right.width());
        debug_assert_eq!(input_left.height(), input_right.height());

        Matrix::new(
            input_left.height(),
            input_left.width(),
            input_left.chain_zip_data(&input_right, |zip| {
                zip.map(|(v_left, v_right)| v_left * v_right).collect()
            }),
        )
    }

    fn grad(
        &self,
        input: (Ref<Matrix>, Ref<Matrix>),
        _: Ref<Matrix>,
        delta: Matrix,
    ) -> (Matrix, Matrix) {
        (
            Matrix::new(
                delta.height(),
                delta.width(),
                input
                    .0
                    .chain_zip_data(&delta, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
            ),
            Matrix::new(
                delta.height(),
                delta.width(),
                input
                    .1
                    .chain_zip_data(&delta, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
            ),
        )
    }
}

impl Mul for Operation {
    type Output = Operation;

    fn mul(self, rhs: Operation) -> Self::Output {
        BinaryOperation::new(self, rhs, MulRunner)
    }
}
