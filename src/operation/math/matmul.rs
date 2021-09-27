use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner, Operation},
};

struct MatrixMultiplicationRunner;

impl BinaryOperationRunner for MatrixMultiplicationRunner {
    fn run(&self, input_left: &Matrix, input_right: &Matrix) -> Matrix {
        debug_assert_eq!(input_left.width(), input_right.height());

        let mut result = Matrix::zeros(input_left.height(), input_right.width());
        for y in 0..input_left.height() {
            for x in 0..input_right.width() {
                let mut val: f32 = 0.0;
                for i in 0..input_left.width() {
                    val += input_left[y][i] * input_right[i][x];
                }
                result[y][x] = val;
            }
        }

        result
    }

    fn grad(&self, child_left: &mut Operation, child_right: &mut Operation, grad: &Matrix) {
        let input_left = child_left.get_output();
        let input_right = child_right.get_output();

        let mut grad_left = Matrix::zeros(input_left.height(), input_left.width());
        for y in 0..grad_left.height() {
            for x in 0..grad_left.width() {
                let mut val: f32 = 0.0;
                for i in 0..grad.width() {
                    val += grad[y][i] * input_right[x][i];
                }
                grad_left[y][x] = val;
            }
        }

        let mut grad_right = Matrix::zeros(input_right.height(), input_right.width());
        for y in 0..grad_right.height() {
            for x in 0..grad_right.width() {
                let mut val: f32 = 0.0;
                for i in 0..grad.height() {
                    val += grad[i][x] * input_left[i][y];
                }
                grad_right[y][x] = val;
            }
        }

        child_left.back_grad(grad_left);
        child_right.back_grad(grad_right);
    }
}

impl Operation {
    pub fn mmul(self, rhs: Operation) -> Self {
        BinaryOperation::new(self, rhs, MatrixMultiplicationRunner)
    }
}
