use crate::{
    matrix::Matrix,
    operation::{Operation, UnaryOperation, UnaryOperationRunner},
};

struct SumRunner;

impl UnaryOperationRunner for SumRunner {
    fn run(&self, input: &Matrix) -> Matrix {
        Matrix::from_const(1, 1, input.chain_data(|data_iter| data_iter.sum::<f32>()))
    }

    fn grad(&self, child: &mut Operation, grad: &Matrix) {
        debug_assert_eq!(grad.get_width(), 1);
        debug_assert_eq!(grad.get_height(), 1);

        let grad_val = grad[0][0];

        let child_in = child.get_output();
        let child_grad = Matrix::new(
            child_in.get_height(),
            child_in.get_width(),
            child_in.chain_data(|child_data| child_data.map(|_| grad_val).collect()),
        );

        child.back_grad(child_grad);
    }
}

impl Operation {
    pub fn sum(self) -> Self {
        UnaryOperation::new(self, SumRunner)
    }
}
