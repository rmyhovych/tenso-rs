use std::ops::Mul;

use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner, Operation},
};

struct MulRunner;

impl BinaryOperationRunner for MulRunner {
    fn run(&self, input_left: &Matrix, input_right: &Matrix) -> Matrix {
        debug_assert_eq!(input_left.get_width(), input_right.get_width());
        debug_assert_eq!(input_left.get_height(), input_right.get_height());

        Matrix::new(
            input_left.get_height(),
            input_left.get_width(),
            input_left.chain_zip_data(input_right, |zip| {
                zip.map(|(v_left, v_right)| v_left * v_right).collect()
            }),
        )
    }

    fn grad(&self, child_left: &mut Operation, child_right: &mut Operation, grad: &Matrix) {
        child_right.back_grad(Matrix::new(
            grad.get_height(),
            grad.get_width(),
            child_left
                .get_output()
                .chain_zip_data(grad, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
        ));
        child_left.back_grad(Matrix::new(
            grad.get_height(),
            grad.get_width(),
            child_right
                .get_output()
                .chain_zip_data(grad, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
        ));
    }
}

impl Mul for Operation {
    type Output = Operation;

    fn mul(self, rhs: Operation) -> Self::Output {
        BinaryOperation::new(self, rhs, MulRunner)
    }
}
