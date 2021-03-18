use super::{BinaryOperation, OperationRef, UnaryOperation};
use crate::matrix::Matrix;

pub fn times(op_input: OperationRef, val: f32) -> OperationRef {
    UnaryOperation::new(
        op_input,
        |input_value| Matrix {
            width: input_value.width,
            height: input_value.height,
            data: input_value.data.iter().map(|val| -val).collect(),
        },
        move |child, grad| {
            child.back_grad(Matrix {
                width: grad.width,
                height: grad.height,
                data: grad.data.iter().map(|v| val * v).collect(),
            });
        },
    )
}

/*
impl std::ops::Add for OperationRef {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Operation::new(
            vec![self, other],
            |input_values| {
                let (left_out, right_out) = (input_values[0], input_values[1]);

                assert_eq!(left_out.width, right_out.width);
                assert_eq!(left_out.height, right_out.height);

                Matrix {
                    width: left_out.width,
                    height: left_out.height,
                    data: left_out
                        .data
                        .iter()
                        .zip(right_out.data.iter())
                        .map(|pair| pair.0 + pair.1)
                        .collect(),
                }
            },
            |children, grad| {
                for child in children {
                    child.back_grad(grad.clone());
                }
            },
        )
    }
}

impl std::ops::Sub for Operation {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + other.times(-1.0)
    }
}

impl std::ops::Mul for Operation {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Operation::new(
            vec![self, other],
            move |input_values: Vec<&Matrix>| {
                let (left_out, right_out) = (input_values[0], input_values[1]);

                assert_eq!(left_out.width, right_out.width);
                assert_eq!(left_out.height, right_out.height);

                Matrix {
                    width: left_out.width,
                    height: left_out.height,
                    data: left_out
                        .data
                        .iter()
                        .zip(right_out.data.iter())
                        .map(|pair| pair.0 * pair.1)
                        .collect(),
                }
            },
            |children, grad| {
                let mut left_child = &children[0];
                let mut right_child = &children[1];
            },
        )
    }
}
*/