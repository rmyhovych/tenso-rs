use tenso_rs::{
    self,
    matrix::Matrix,
    operation::input::InputPlaceholder,
    optim::{Optimizer, OptimizerRunner, RunningOptimizer},
};

#[test]
fn run() {
    let mat0 = Matrix::randn(3, 5, 0.0, 1.0);
    let placeholder0 = InputPlaceholder::with_value(mat0.clone());

    let mat1 = Matrix::randn(3, 5, 0.0, 1.0);
    let placeholder1 = InputPlaceholder::with_value(mat1.clone());

    let mut result_op = placeholder0.clone() + placeholder1.clone();
    let result = result_op.run();

    for y in 0..result.get_height() {
        for x in 0..result.get_width() {
            assert_eq!(mat0[y][x] + mat1[y][x], result[y][x]);
        }
    }
}

struct TestOptimizer {
    expected_grads: Vec<Matrix>,
}

impl OptimizerRunner for TestOptimizer {
    fn run(&mut self, variables: Vec<(&mut Matrix, &mut Matrix)>) {
        for ((_, grad), expected_grad) in variables.into_iter().zip(&self.expected_grads) {
            assert_eq!(expected_grad.get_height(), grad.get_height());
            assert_eq!(expected_grad.get_width(), grad.get_width());

            for y in 0..grad.get_height() {
                for x in 0..grad.get_width() {
                    assert_eq!(expected_grad[y][x], grad[y][x]);
                }
            }
        }
    }
}

#[test]
fn back() {
    let width = 3;
    let height = 5;

    let mat0 = Matrix::randn(height, width, 0.0, 1.0);
    let var0 = mat0.clone().as_variable();

    let mat1 = Matrix::randn(height, width, 0.0, 1.0);
    let var1 = mat1.clone().as_variable();

    let mut result_op = (var0.clone() + var1.clone()).sum();
    result_op.run();
    result_op.back();

    let mut optim = RunningOptimizer::new(TestOptimizer {
        expected_grads: vec![
            Matrix::from_const(height, width, 1.0),
            Matrix::from_const(height, width, 1.0),
        ],
    });
    var1.add_to_optimizer(&mut optim);
    var0.add_to_optimizer(&mut optim);

    optim.step();
}
