use std::{cell::Ref, ops::Add};

use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner, Operation},
};

struct AddRunner;

impl BinaryOperationRunner for AddRunner {
    fn run(&self, input_left: Ref<Matrix>, input_right: Ref<Matrix>) -> Matrix {
        debug_assert_eq!(input_left.width(), input_right.width());
        debug_assert_eq!(input_left.height(), input_right.height());

        let out_data = input_left.chain_zip_data(&input_right, |zip| {
            zip.map(|(v_left, v_right)| v_left + v_right).collect()
        });

        Matrix::new(input_left.height(), input_left.width(), out_data)
    }

    fn grad(
        &self,
        _: (Ref<Matrix>, Ref<Matrix>),
        _: Ref<Matrix>,
        delta: Matrix,
    ) -> (Matrix, Matrix) {
        (delta.clone(), delta.clone())
    }
}

impl Add for Operation {
    type Output = Operation;

    fn add(self, rhs: Operation) -> Self::Output {
        BinaryOperation::new(self, rhs, AddRunner)
    }
}
