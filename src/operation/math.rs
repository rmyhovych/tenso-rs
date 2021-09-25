use crate::matrix::Matrix;

use super::{Operation, UnaryOperation};

fn sigmoid(val: f32) -> f32 {
    1.0 / (1.0 + (-val).exp())
}

fn sigmoid_prime(val: f32) -> f32 {
    let sig = sigmoid(val);
    sig * (1.0 - sig)
}

impl Operation {
    pub fn sigmoid(self) -> Self {
        UnaryOperation::new(
            self,
            |input_value| {
                Matrix::new(
                    input_value.get_height(),
                    input_value.get_width(),
                    input_value.chain_data(|data_iter| data_iter.map(|v| sigmoid(*v)).collect()),
                )
            },
            |child, grad| {
                let child_in = child.get_output();
                let child_grad = Matrix::new(
                    child_in.get_height(),
                    child_in.get_width(),
                    child_in.chain_zip_data(grad, |child_data| {
                        child_data.map(|(ci, gr)| gr * sigmoid_prime(*ci)).collect()
                    }),
                );

                child.back_grad(child_grad);
            },
        )
    }

    pub fn relu(self) -> Self {
        UnaryOperation::new(
            self,
            move |input_value| {
                Matrix::new(
                    input_value.get_height(),
                    input_value.get_width(),
                    input_value.chain_data(|data_iter| data_iter.map(|v| if *v > 0.0 { *v } else { 0.0 }).collect()),
                )
            },
            move |child, grad| {
                let child_in = child.get_output();
                let child_grad = Matrix::new(
                    child_in.get_height(),
                    child_in.get_width(),
                    child_in.chain_zip_data(grad, |child_data| {
                        child_data.map(|(ci, gr)| if *ci > 0.0 { *gr } else { 0.0 } ).collect()
                    }),
                );

                child.back_grad(child_grad);
            },
        )
    }

    pub fn pow(self, power: f32) -> Self {
        UnaryOperation::new(
            self,
            move |input_value| {
                Matrix::new(
                    input_value.get_height(),
                    input_value.get_width(),
                    input_value.chain_data(|data_iter| data_iter.map(|v| v.powf(power)).collect()),
                )
            },
            move |child, grad| {
                let child_in = child.get_output();
                let child_grad = Matrix::new(
                    child_in.get_height(),
                    child_in.get_width(),
                    child_in.chain_zip_data(grad, |child_data| {
                        child_data.map(|(ci, gr)| gr * power * ci.powf(power - 1.0)).collect()
                    }),
                );

                child.back_grad(child_grad);
            },
        )
    }
}
