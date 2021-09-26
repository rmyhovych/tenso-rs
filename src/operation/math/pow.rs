use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct PowRunner {
    power: f32,
}

impl UnaryOperationRunner for PowRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::new(
            input.get_height(),
            input.get_width(),
            input.chain_data(|data_iter| data_iter.map(|v| v.powf(self.power)).collect()),
        )
    }

    fn grad(&self, child: &mut Operation, grad: &Matrix) {
        let child_in = child.get_output();
        let child_grad = Matrix::new(
            child_in.get_height(),
            child_in.get_width(),
            child_in.chain_zip_data(grad, |child_data| {
                child_data
                    .map(|(ci, gr)| gr * self.power * ci.powf(self.power - 1.0))
                    .collect()
            }),
        );

        child.back_grad(child_grad);
    }
}

impl Operation {
    pub fn pow(self, power: f32) -> Self {
        UnaryOperation::new(self, PowRunner { power })
    }
}
