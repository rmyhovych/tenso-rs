use crate::{
    matrix::{Matrix, MatrixRef},
    optim::Optimizer,
};
use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

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

    pub fn run(&mut self) -> MatrixRef {
        self.op.borrow_mut().run()
    }

    pub fn back(&mut self) {
        self.op.borrow_mut().back();
    }

    pub fn get_output(&self) -> MatrixRef {
        self.op.borrow().get_output()
    }

    pub fn set_input(&mut self, input: MatrixRef) {
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
    fn run(&mut self) -> MatrixRef;

    fn back(&mut self) {
        debug_assert!(
            self.get_output().get().height() == 1 && self.get_output().get().width() == 1,
            "Cant backpropagate a non-unit matrix!"
        );

        self.back_grad(Matrix::from_const(1, 1, 1.0));
    }

    fn back_grad(&mut self, delta: Matrix);

    fn get_output(&self) -> MatrixRef;

    fn set_input(&mut self, input: MatrixRef);

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer);
}

/*---- UNARY -------------------------------------------------------------------------------------*/

trait UnaryOperationRunner {
    fn run(&self, input: Ref<Matrix>) -> Matrix;

    fn grad(&self, input: Ref<Matrix>, output: Ref<Matrix>, delta: Matrix) -> Matrix;
}

struct UnaryOperation<R: UnaryOperationRunner + 'static> {
    op_input: Operation,
    output: MatrixRef,

    runner: R,
}

impl<R: UnaryOperationRunner + 'static> UnaryOperation<R> {
    fn new(op_input: Operation, runner: R) -> Operation {
        Operation::new(Self {
            op_input,
            output: MatrixRef::new(Matrix::zeros(0, 0)),
            runner,
        })
    }
}

impl<R: UnaryOperationRunner> OperationBase for UnaryOperation<R> {
    fn run(&mut self) -> MatrixRef {
        let out_input = self.op_input.run();
        self.output.set(self.runner.run(out_input.get()));
        self.output.clone()
    }

    fn back_grad(&mut self, delta: Matrix) {
        let input = self.op_input.get_output();
        let delta_input = self.runner.grad(input.get(), self.output.get(), delta);
        self.op_input.back_grad(delta_input);
    }

    fn get_output(&self) -> MatrixRef {
        self.output.clone()
    }

    fn set_input(&mut self, _: MatrixRef) {}

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer) {
        self.op_input.add_to_optimizer(optim);
    }
}

/*---- BINARY ------------------------------------------------------------------------------------*/

trait BinaryOperationRunner {
    fn run(&self, input_left: Ref<Matrix>, input_right: Ref<Matrix>) -> Matrix;

    fn grad(
        &self,
        input: (Ref<Matrix>, Ref<Matrix>),
        output: Ref<Matrix>,
        delta: Matrix,
    ) -> (Matrix, Matrix);
}

struct BinaryOperation<R: BinaryOperationRunner + 'static> {
    op_left: Operation,
    op_right: Operation,

    output: MatrixRef,

    runner: R,
}

impl<R: BinaryOperationRunner + 'static> BinaryOperation<R> {
    fn new(op_left: Operation, op_right: Operation, runner: R) -> Operation {
        Operation::new(Self {
            op_left: op_left,
            op_right: op_right,

            output: MatrixRef::new(Matrix::zeros(0, 0)),
            runner,
        })
    }
}

impl<R: BinaryOperationRunner + 'static> OperationBase for BinaryOperation<R> {
    fn run(&mut self) -> MatrixRef {
        let out_left = self.op_left.run();
        let out_right = self.op_right.run();

        self.output
            .set(self.runner.run(out_left.get(), out_right.get()));
        self.output.clone()
    }

    fn back_grad(&mut self, grad: Matrix) {
        let input_left = self.op_left.get_output();
        let input_right = self.op_right.get_output();
        let (delta_left, delta_right) = self.runner.grad(
            (input_left.get(), input_right.get()),
            self.output.get(),
            grad,
        );

        self.op_left.back_grad(delta_left);
        self.op_right.back_grad(delta_right);
    }

    fn get_output(&self) -> MatrixRef {
        self.output.clone()
    }

    fn set_input(&mut self, _: MatrixRef) {}

    fn add_to_optimizer(&self, optim: &mut dyn Optimizer) {
        self.op_left.add_to_optimizer(optim);
        self.op_right.add_to_optimizer(optim);
    }
}

/*------------------------------------------------------------------------------------------------*/
