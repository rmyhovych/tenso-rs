use tenso_rs::{
    self,
    matrix::Matrix,
    operation::input::InputPlaceholder,
    optim::{Optimizer, OptimizerRunner, RunningOptimizer},
};

#[test]
fn run() {
    let mat0 = Matrix::new(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let placeholder0 = InputPlaceholder::with_value(mat0.clone());

    let mat1 = Matrix::new(2, 2, vec![6.0, 7.0, 8.0, 9.0]);
    let placeholder1 = InputPlaceholder::with_value(mat1.clone());

    let mut result_op = placeholder0.clone().mmul(placeholder1.clone());

    let result = result_op.run();
    let expected_result = Matrix::new(3, 2, vec![8.0, 9.0, 36.0, 41.0, 64.0, 73.0]);
    for y in 0..result.height() {
        for x in 0..result.width() {
            assert_eq!(expected_result[y][x], result[y][x]);
        }
    }
}

struct TestOptimizer {
    expected_grads: Vec<Matrix>,
}

impl OptimizerRunner for TestOptimizer {
    fn run(&mut self, variables: Vec<(&mut Matrix, &mut Matrix)>) {
        for ((_, grad), expected_grad) in variables.into_iter().zip(&self.expected_grads) {
            assert_eq!(expected_grad.height(), grad.height());
            assert_eq!(expected_grad.width(), grad.width());

            for y in 0..grad.height() {
                for x in 0..grad.width() {
                    assert_eq!(expected_grad[y][x], grad[y][x]);
                }
            }
        }
    }
}

#[test]
fn back0() {
    let mat0 = Matrix::new(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let var0 = mat0.clone().as_variable();

    let mat1 = Matrix::new(2, 2, vec![6.0, 7.0, 8.0, 9.0]);
    let var1 = mat1.clone().as_variable();

    let mut result_op = var0.clone().mmul(var1.clone()).sum();

    result_op.run();
    result_op.back();

    let mut optim = RunningOptimizer::new(TestOptimizer {
        expected_grads: vec![
            Matrix::new(3, 2, vec![13.0, 17.0, 13.0, 17.0, 13.0, 17.0]),
            Matrix::new(2, 2, vec![6.0, 6.0, 9.0, 9.0]),
        ],
    });

    var0.add_to_optimizer(&mut optim);
    var1.add_to_optimizer(&mut optim);

    optim.step();
}

#[test]
fn back1() {
    let mat0 = Matrix::new(3, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let var0 = mat0.clone().as_variable();

    let mat1 = Matrix::new(3, 2, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    let var1 = mat1.clone().as_variable();

    let mut result_op = var0.clone().mmul(var1.clone()).sum();

    result_op.run();
    result_op.back();

    let mut optim = RunningOptimizer::new(TestOptimizer {
        expected_grads: vec![
            Matrix::new(
                3,
                3,
                vec![13.0, 17.0, 21.0, 13.0, 17.0, 21.0, 13.0, 17.0, 21.0],
            ),
            Matrix::new(3, 2, vec![9.0, 9.0, 12.0, 12.0, 15.0, 15.0]),
        ],
    });

    var0.add_to_optimizer(&mut optim);
    var1.add_to_optimizer(&mut optim);

    optim.step();
}
