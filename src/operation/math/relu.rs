use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct ReluRunner;

impl UnaryOperationRunner for ReluRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::new(
            input.height(),
            input.width(),
            input.chain_data(|data_iter| {
                data_iter.map(|v| if *v > 0.0 { *v } else { 0.0 }).collect()
            }),
        )
    }

    fn grad(&self, child: &mut Operation, grad: &Matrix) {
        let child_in = child.get_output();
        let child_grad = Matrix::new(
            child_in.height(),
            child_in.width(),
            child_in.chain_zip_data(grad, |child_data| {
                child_data
                    .map(|(ci, gr)| if *ci > 0.0 { *gr } else { 0.0 })
                    .collect()
            }),
        );

        child.back_grad(child_grad);
    }
}

impl Operation {
    pub fn relu(self) -> Self {
        UnaryOperation::new(self, ReluRunner)
    }
}
