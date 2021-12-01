use std::ops::Add;

use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner, Operation},
};

use super::OperationRef;

pub struct AddRunner;

impl BinaryOperationRunner for AddRunner {
    fn run(&self, input_left: &Matrix, input_right: &Matrix) -> Matrix {
        debug_assert_eq!(input_left.width(), input_right.width());
        debug_assert_eq!(input_left.height(), input_right.height());

        let out_data = input_left.chain_zip_data(input_right, |zip| {
            zip.map(|(v_left, v_right)| v_left + v_right).collect()
        });

        Matrix::new(input_left.height(), input_left.width(), out_data)
    }

    fn grad(
        &self,
        _input_left: &Matrix,
        _input_right: &Matrix,
        delta: &Matrix,
    ) -> (Matrix, Matrix) {
        (delta.clone(), delta.clone())
    }
}

impl Add for OperationRef {
    type Output = OperationRef;

    fn add(self, rhs: OperationRef) -> Self::Output {
        BinaryOperation::new(&self, &rhs, AddRunner)
    }
}
