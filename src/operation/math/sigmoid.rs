use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct SigmoidRunner;

impl SigmoidRunner {
    fn sigmoid(val: f32) -> f32 {
        1.0 / (1.0 + (-val).exp())
    }

    fn sigmoid_prime(val: f32) -> f32 {
        let sig = Self::sigmoid(val);
        sig * (1.0 - sig)
    }
}

impl UnaryOperationRunner for SigmoidRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::new(
            input.get_height(),
            input.get_width(),
            input.chain_data(|data_iter| data_iter.map(|v| Self::sigmoid(*v)).collect()),
        )
    }

    fn grad(&self, child: &mut Operation, grad: &Matrix) {
        let child_in = child.get_output();
        let child_grad = Matrix::new(
            child_in.get_height(),
            child_in.get_width(),
            child_in.chain_zip_data(grad, |child_data| {
                child_data
                    .map(|(ci, gr)| gr * Self::sigmoid_prime(*ci))
                    .collect()
            }),
        );

        child.back_grad(child_grad);
    }
}

impl Operation {
    pub fn sigmoid(self) -> Self {
        UnaryOperation::new(self, SigmoidRunner)
    }
}
