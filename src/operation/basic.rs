use std::{
    ops::{Add, Mul, Sub},
    slice::Iter,
};

use super::{BinaryOperation, Operation, UnaryOperation};
use crate::matrix::Matrix;

impl Operation {
    pub fn sum(self) -> Operation {
        UnaryOperation::new(
            self,
            |input_value| {
                Matrix::from_const(
                    1,
                    1,
                    input_value.chain_data(|data_iter| data_iter.sum::<f32>()),
                )
            },
            move |child, grad| {
                assert_eq!(grad.get_width(), 1);
                assert_eq!(grad.get_height(), 1);

                let grad_val = grad.get_value(0, 0);

                let child_in = child.get_output();
                let child_grad = Matrix::new(
                    child_in.get_height(),
                    child_in.get_width(),
                    child_in.chain_data(|child_data| child_data.map(|_| grad_val).collect()),
                );

                child.back_grad(child_grad);
            },
        )
    }

    pub fn mean(self) -> Operation {
        UnaryOperation::new(
            self,
            |input_value| {
                Matrix::from_const(
                    1,
                    1,
                    input_value.chain_data(|data_iter: Iter<f32>| data_iter.sum::<f32>())
                        / (input_value.get_width() * input_value.get_height()) as f32,
                )
            },
            move |child, grad| {
                assert_eq!(grad.get_width(), 1);
                assert_eq!(grad.get_height(), 1);

                let child_in = child.get_output();
                let mat_size = (child_in.get_width() * child_in.get_height()) as f32;

                let grad_val = grad.get_value(0, 0);

                let child_in = child.get_output();
                let child_grad = Matrix::new(
                    child_in.get_height(),
                    child_in.get_width(),
                    child_in
                        .chain_data(|child_data| child_data.map(|_| grad_val / mat_size).collect()),
                );

                child.back_grad(child_grad);
            },
        )
    }

    pub fn times(self, val: f32) -> Operation {
        UnaryOperation::new(
            self,
            move |input_value| {
                Matrix::new(
                    input_value.get_height(),
                    input_value.get_width(),
                    input_value
                        .chain_data(|data_iter| data_iter.map(|mat_val| val * mat_val).collect()),
                )
            },
            move |child, grad| {
                child.back_grad(Matrix::new(
                    grad.get_height(),
                    grad.get_width(),
                    grad.chain_data(|di| di.map(|v| val * v).collect()),
                ));
            },
        )
    }

    pub fn cross(self, other: Operation) -> Operation {
        BinaryOperation::new(
            self,
            other,
            |in_left, in_right| {
                assert_eq!(in_left.get_width(), in_right.get_height());

                let mut result = Matrix::zeros(in_left.get_height(), in_right.get_width());
                for y in 0..in_left.get_height() {
                    for x in 0..in_right.get_width() {
                        let mut val: f32 = 0.0;
                        for i in 0..in_left.get_width() {
                            val += in_left.get_value(y, i) * in_right.get_value(i, x);
                        }
                        result.set_value(val, y, x);
                    }
                }

                result
            },
            |op_left, op_right, grad| {
                let input_left = op_left.get_output();
                let input_right = op_right.get_output();

                let mut grad_left = Matrix::zeros(input_left.get_height(), input_left.get_width());
                for y in 0..grad_left.get_height() {
                    for x in 0..grad_left.get_width() {
                        let mut val: f32 = 0.0;
                        for i in 0..grad.get_width() {
                            val += grad.get_value(y, i) * input_right.get_value(x, i);
                        }
                        grad_left.set_value(val, y, x);
                    }
                }

                let mut grad_right =
                    Matrix::zeros(input_right.get_height(), input_right.get_width());
                for y in 0..grad_right.get_height() {
                    for x in 0..grad_right.get_width() {
                        let mut val: f32 = 0.0;
                        for i in 0..grad.get_height() {
                            val += grad.get_value(i, x) * input_left.get_value(i, y);
                        }
                        grad_right.set_value(val, y, x);
                    }
                }

                op_left.back_grad(grad_left);
                op_right.back_grad(grad_right);
            },
        )
    }
}

impl Add for Operation {
    type Output = Operation;

    fn add(self, rhs: Operation) -> Self::Output {
        BinaryOperation::new(
            self,
            rhs,
            |in_left, in_right| {
                assert_eq!(in_left.get_width(), in_right.get_width());
                assert_eq!(in_left.get_height(), in_right.get_height());

                let out_data = in_left.chain_zip_data(in_right, |zip| {
                    zip.map(|(v_left, v_right)| v_left + v_right).collect()
                });

                Matrix::new(in_left.get_height(), in_left.get_width(), out_data)
            },
            |op_left, op_right, grad| {
                op_left.back_grad(grad.clone());
                op_right.back_grad(grad.clone());
            },
        )
    }
}

impl Sub for Operation {
    type Output = Operation;

    fn sub(self, rhs: Operation) -> Self::Output {
        self + rhs.times(-1.0)
    }
}

impl Mul for Operation {
    type Output = Operation;

    fn mul(self, rhs: Operation) -> Self::Output {
        BinaryOperation::new(
            self,
            rhs,
            |in_left, in_right| {
                assert_eq!(in_left.get_width(), in_right.get_width());
                assert_eq!(in_left.get_height(), in_right.get_height());

                Matrix::new(
                    in_left.get_height(),
                    in_left.get_width(),
                    in_left.chain_zip_data(in_right, |zip| {
                        zip.map(|(v_left, v_right)| v_left * v_right).collect()
                    }),
                )
            },
            |op_left, op_right, grad| {
                op_right.back_grad(Matrix::new(
                    grad.get_height(),
                    grad.get_width(),
                    op_left
                        .get_output()
                        .chain_zip_data(grad, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
                ));
                op_left.back_grad(Matrix::new(
                    grad.get_height(),
                    grad.get_width(),
                    op_right
                        .get_output()
                        .chain_zip_data(grad, |zip| zip.map(|(v_grad, v)| v_grad * v).collect()),
                ));
            },
        )
    }
}
