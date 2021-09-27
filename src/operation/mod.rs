use crate::{matrix::Matrix, optim::Optimizer};
use std::{cell::RefCell, rc::Rc};

pub mod input;
pub mod math;

/*------------------------------------------------------------------------------------------------*/

pub struct Operation {
    op: Rc<RefCell<dyn OperationBase>>,
}

impl Operation {
    fn new(op: impl OperationBase + 'static) -> Self {
        Self {
            op: Rc::new(RefCell::new(op)),
        }
    }

    pub fn run(&mut self) -> Matrix {
        self.op.borrow_mut().run()
    }

    pub fn back(&mut self) {
        self.op.borrow_mut().back();
    }

    pub fn get_output(&self) -> Matrix {
        self.op.borrow().get_output()
    }

    pub fn set_input(&mut self, input: Matrix) {
        self.op.borrow_mut().set_input(input);
    }

    pub fn add_to_optimizer(&self, optim: &mut dyn Optimizer) {
        self.op.borrow_mut().add_to_optimizer(optim);
    }

    /*------------------------------------------------------*/

    fn back_grad(&mut self, grad: Matrix) {
        self.op.borrow_mut().back_grad(grad);
    }
}

impl Clone for Operation {
    fn clone(&self) -> Self {
        Operation {
            op: Rc::clone(&self.op),
        }
    }
}

/*------------------------------------------------------------------------------------------------*/

trait OperationBase {
    fn run(&mut self) -> Matrix;

    fn back(&mut self);

    fn back_grad(&mut self, grad: Matrix);

    fn get_output(&self) -> Matrix;

    fn set_input(&mut self, input: Matrix);

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer);
}

/*------------------------------------------------------------------------------------------------*/

trait UnaryOperationRunner {
    fn run(&self, input: &Matrix) -> Matrix;

    fn grad(&self, child: &mut Operation, grad: &Matrix);
}

struct UnaryOperation<R: UnaryOperationRunner + 'static> {
    op_input: Operation,
    output: Matrix,

    runner: R,
}

impl<R: UnaryOperationRunner + 'static> UnaryOperation<R> {
    fn new(op_input: Operation, runner: R) -> Operation {
        Operation::new(Self {
            op_input,
            output: Matrix::zeros(0, 0),
            runner,
        })
    }
}

impl<R: UnaryOperationRunner> OperationBase for UnaryOperation<R> {
    fn run(&mut self) -> Matrix {
        let out_input = self.op_input.run();
        self.output = self.runner.run(&out_input);

        self.output.clone()
    }

    fn back(&mut self) {
        debug_assert!(
            self.output.get_height() == 1 && self.output.get_width() == 1,
            "Cant backpropagate a non-unit matrix!"
        );

        let grad = Matrix::from_const(1, 1, 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.runner.grad(&mut self.op_input, &grad);
    }

    fn get_output(&self) -> Matrix {
        self.output.clone()
    }

    fn set_input(&mut self, _: Matrix) {}

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer) {
        self.op_input.add_to_optimizer(optim);
    }
}

/*------------------------------------------------------------------------------------------------*/

trait BinaryOperationRunner {
    fn run(&self, input_left: &Matrix, input_right: &Matrix) -> Matrix;

    fn grad(&self, child_left: &mut Operation, child_right: &mut Operation, gradient: &Matrix);
}

struct BinaryOperation<R: BinaryOperationRunner + 'static> {
    op_left: Operation,
    op_right: Operation,

    output: Matrix,

    runner: R,
}

impl<R: BinaryOperationRunner + 'static> BinaryOperation<R> {
    fn new(op_left: Operation, op_right: Operation, runner: R) -> Operation {
        Operation::new(Self {
            op_left: op_left,
            op_right: op_right,

            output: Matrix::zeros(0, 0),
            runner,
        })
    }
}

impl<R: BinaryOperationRunner + 'static> OperationBase for BinaryOperation<R> {
    fn run(&mut self) -> Matrix {
        let out_left = self.op_left.run();
        let out_right = self.op_right.run();

        self.output = self.runner.run(&out_left, &out_right);

        self.output.clone()
    }

    fn back(&mut self) {
        debug_assert!(
            self.output.get_height() == 1 && self.output.get_width() == 1,
            "Cant backpropagate a non-unit matrix!"
        );

        let grad = Matrix::from_const(1, 1, 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.runner
            .grad(&mut self.op_left, &mut self.op_right, &grad);
    }

    fn get_output(&self) -> Matrix {
        self.output.clone()
    }

    fn set_input(&mut self, _: Matrix) {}

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer) {
        self.op_left.add_to_optimizer(optim);
        self.op_right.add_to_optimizer(optim);
    }
}

/*------------------------------------------------------------------------------------------------*/
