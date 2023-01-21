use std::ops::Deref;

use crate::matrix::{Matrix, MatrixData};

pub mod math;

pub trait Operation {
    fn back_grad(&mut self, delta: MatrixData);
}

/*------------------------------------------------------------------------------------------------*/

trait UnaryOperationRunner {
    fn run(&self, input: &MatrixData) -> MatrixData;

    fn grad(&self, input: &MatrixData, delta: &MatrixData) -> MatrixData;
}

struct UnaryOperation<R: UnaryOperationRunner + 'static> {
    input: Matrix,
    runner: R,
}

impl<R: UnaryOperationRunner + 'static> UnaryOperation<R> {
    fn new(input: Matrix, runner: R) -> Matrix {
        let op = Self { input, runner };
        let output_data = op.runner.run(op.input.data().deref());
        Matrix::new_output(output_data, op)
    }
}

impl<R: UnaryOperationRunner> Operation for UnaryOperation<R> {
    fn back_grad(&mut self, delta: MatrixData) {
        let new_delta = self.runner.grad(self.input.data().deref(), &delta);
        self.input.
    }
}

/*------------------------------------------------------------------------------------------------*/

trait BinaryOperationRunner {
    fn run(&self, input_left: &MatrixData, input_right: &MatrixData) -> MatrixData;

    fn grad(
        &self,
        input_left: &MatrixData,
        input_right: &MatrixData,
        delta: &MatrixData,
    ) -> (MatrixData, MatrixData);
}

struct BinaryOperation<R: BinaryOperationRunner + 'static> {
    input_left: Matrix,
    input_right: Matrix,
    runner: R,
}

impl<R: BinaryOperationRunner + 'static> BinaryOperation<R> {
    fn new(input_left: Matrix, input_right: Matrix, runner: R) -> Matrix {
        let op = Self {
            input_left,
            input_right,
            runner,
        };

        let output_data = op
            .runner
            .run(op.input_left.data().deref(), op.input_right.data().deref());

        Matrix::new_output(output_data, op)
    }
}

impl<R: BinaryOperationRunner> Operation for BinaryOperation<R> {
    fn back_grad(&mut self, delta: MatrixData) {}
}

/*------------------------------------------------------------------------------------------------*/
