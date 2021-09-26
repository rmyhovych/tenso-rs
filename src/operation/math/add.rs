use std::ops::Add;

use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner, Operation},
};

struct AddRunner;

impl BinaryOperationRunner for AddRunner {
    fn run(&self, input_left: &Matrix, input_right: &Matrix) -> Matrix {
        assert_eq!(input_left.get_width(), input_right.get_width());
        assert_eq!(input_left.get_height(), input_right.get_height());

        let out_data = input_left.chain_zip_data(input_right, |zip| {
            zip.map(|(v_left, v_right)| v_left + v_right).collect()
        });

        Matrix::new(input_left.get_height(), input_left.get_width(), out_data)
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
