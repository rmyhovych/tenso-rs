use std::cell::Ref;

use crate::{
    matrix::Matrix,
    operation::{BinaryOperation, BinaryOperationRunner, Operation},
};

struct MatrixMultiplicationRunner;

impl BinaryOperationRunner for MatrixMultiplicationRunner {
    fn run(&self, input_left: Ref<Matrix>, input_right: Ref<Matrix>) -> Matrix {
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

    fn grad(
        &self,
        input: (Ref<Matrix>, Ref<Matrix>),
        _: Ref<Matrix>,
        delta: Matrix,
    ) -> (Matrix, Matrix) {
        let mut delta_left = Matrix::zeros(input.0.height(), input.0.width());
        for y in 0..delta_left.height() {
            for x in 0..delta_left.width() {
                let mut val: f32 = 0.0;
                for i in 0..delta.width() {
                    val += delta[y][i] * input.1[x][i];
                }
                delta_left[y][x] = val;
            }
        }

        let mut delta_right = Matrix::zeros(input.1.height(), input.1.width());
        for y in 0..delta_right.height() {
            for x in 0..delta_right.width() {
                let mut val: f32 = 0.0;
                for i in 0..delta.height() {
                    val += delta[i][x] * input.0[i][y];
                }
                delta_right[y][x] = val;
            }
        }

        (delta_left, delta_right)
    }
}

impl Operation {
    pub fn mmul(self, rhs: Operation) -> Self {
        BinaryOperation::new(self, rhs, MatrixMultiplicationRunner)
    }
}
