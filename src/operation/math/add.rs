use std::ops::Add;

use crate::{matrix::Matrix, operation::BinaryOperationRunner};

struct AddRunner;

impl BinaryOperationRunner for AddRunner {
    fn run(&self, input_left: &MatrixData, input_right: &MatrixData) -> MatrixData {
        debug_assert_eq!(input_left.width(), input_right.width());
        debug_assert_eq!(input_left.height(), input_right.height());

        let out_data = input_left.chain_zip_data(input_right, |zip| {
            zip.map(|(v_left, v_right)| v_left + v_right).collect()
        });

        Matrix::new(input_left.height(), input_left.width(), out_data)
    }

    fn grad(&self, child_left: &mut Operation, child_right: &mut Operation, grad: &Matrix) {
        child_left.back_grad(grad.clone());
        child_right.back_grad(grad.clone());
    }
}

impl Add for Operation {
    type Output = Operation;

    fn add(self, rhs: Operation) -> Self::Output {
        BinaryOperation::new(self, rhs, AddRunner)
    }
}
