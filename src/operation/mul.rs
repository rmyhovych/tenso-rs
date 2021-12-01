use std::ops::Mul;

use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner},
};

use super::OperationRef;

struct MulRunner;

impl BinaryOperationRunner for MulRunner {
    fn run(&self, input_left: &Matrix, input_right: &Matrix) -> Matrix {
        debug_assert_eq!(input_left.width(), input_right.width());
        debug_assert_eq!(input_left.height(), input_right.height());

        Matrix::new(
            input_left.height(),
            input_left.width(),
            input_left.chain_zip_data(input_right, |zip| {
                zip.map(|(v_left, v_right)| v_left * v_right).collect()
            }),
        )
    }

    fn grad(&self, input_left: &Matrix, input_right: &Matrix, delta: &Matrix) -> (Matrix, Matrix) {
        let delta_left = Matrix::new(
            delta.height(),
            delta.width(),
            input_left.chain_zip_data(delta, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
        );

        let delta_right = Matrix::new(
            delta.height(),
            delta.width(),
            input_right.chain_zip_data(delta, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
        );

        (delta_left, delta_right)
    }
}

impl Mul for OperationRef {
    type Output = OperationRef;

    fn mul(self, rhs: OperationRef) -> Self::Output {
        BinaryOperation::new(&self, &rhs, MulRunner)
    }
}
