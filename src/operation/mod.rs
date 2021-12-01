use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
};

use crate::matrix::Matrix;

pub mod add;
pub mod matmul;
pub mod mean;
pub mod mul;
pub mod pow;
pub mod relu;
pub mod sigmoid;
pub mod sub;
pub mod sum;
pub mod times;

/*------------------------------------------------------------------------------------------------*/

pub struct OperationRef {
    op: Rc<RefCell<dyn Operation>>,
}

impl OperationRef {
    fn new(op: impl Operation + 'static) -> Self {
        Self {
            op: Rc::new(RefCell::new(op)),
        }
    }

    pub fn back(&mut self) {
        let mut op_ref = self.as_mut();
        let value = op_ref.get_value();
        debug_assert!(
            value.height() == 1 && value.width() == 1,
            "Cant backpropagate a non-unit matrix!"
        );

        let grad = Matrix::from_const(1, 1, 1.0);
        op_ref.back_grad(grad);
    }

    pub fn as_ref(&self) -> Ref<dyn Operation> {
        self.op.borrow()
    }

    pub fn as_mut(&mut self) -> RefMut<dyn Operation> {
        self.op.borrow_mut()
    }
}

impl Clone for OperationRef {
    fn clone(&self) -> Self {
        Self {
            op: Rc::clone(&self.op),
        }
    }
}

/*------------------------------------------------------------------------------------------------*/

pub trait Operation {
    fn back_grad(&mut self, delta: Matrix);

    fn get_value(&self) -> &Matrix;

    fn as_variable(&mut self) -> Option<&mut Variable> {
        None
    }
}

/*------------------------------------------------------------------------------------------------*/

pub struct Constant {
    value: Matrix,
}

impl Constant {
    fn new(value: Matrix) -> OperationRef {
        OperationRef::new(Self { value })
    }
}

impl Operation for Constant {
    fn back_grad(&mut self, _delta: Matrix) {}

    fn get_value(&self) -> &Matrix {
        &self.value
    }
}

impl Matrix {
    pub fn to_const(self) -> OperationRef {
        Constant::new(self)
    }
}

/*------------------------------------------------------------------------------------------------*/

pub struct Variable {
    pub value: Matrix,
    pub grad: Matrix,
}

impl Variable {
    pub fn new(value: Matrix) -> OperationRef {
        let width = value.width();
        let height = value.height();
        OperationRef::new(Self {
            value,
            grad: Matrix::zeros(width, height),
        })
    }

    pub fn zero_grad(&mut self) {
        self.grad.clear();
    }
}

impl Operation for Variable {
    fn back_grad(&mut self, delta: Matrix) {
        self.grad = if delta.width() == self.grad.width() && delta.height() == self.grad.height() {
            Matrix::new(
                delta.height(),
                delta.width(),
                delta.chain_zip_data(&self.grad, |data_zip| {
                    data_zip.map(|(v0, v1)| v0 + v1).collect()
                }),
            )
        } else {
            delta
        };
    }

    fn get_value(&self) -> &Matrix {
        &self.value
    }

    fn as_variable(&mut self) -> Option<&mut Variable> {
        Some(self)
    }
}

/*------------------------------------------------------------------------------------------------*/

pub trait UnaryOperationRunner {
    fn run(&self, input: &Matrix) -> Matrix;

    fn grad(&self, input: &Matrix, delta: &Matrix) -> Matrix;
}

pub struct UnaryOperation<R: UnaryOperationRunner + 'static> {
    input: OperationRef,
    value: Matrix,

    runner: R,
}

impl<R: UnaryOperationRunner + 'static> UnaryOperation<R> {
    fn new(input: &OperationRef, runner: R) -> OperationRef {
        let value = runner.run(input.as_ref().get_value());
        OperationRef::new(Self {
            input: input.clone(),
            value,
            runner,
        })
    }
}

impl<R: UnaryOperationRunner> Operation for UnaryOperation<R> {
    fn back_grad(&mut self, delta: Matrix) {
        let mut input_ref = self.input.as_mut();
        let new_delta = self.runner.grad(input_ref.get_value(), &delta);
        input_ref.back_grad(new_delta);
    }

    fn get_value(&self) -> &Matrix {
        &self.value
    }
}

/*------------------------------------------------------------------------------------------------*/

pub trait BinaryOperationRunner {
    fn run(&self, input_left: &Matrix, input_right: &Matrix) -> Matrix;

    fn grad(&self, input_left: &Matrix, input_right: &Matrix, delta: &Matrix) -> (Matrix, Matrix);
}

pub struct BinaryOperation<R: BinaryOperationRunner + 'static> {
    input_left: OperationRef,
    input_right: OperationRef,

    value: Matrix,

    runner: R,
}

impl<R: BinaryOperationRunner + 'static> BinaryOperation<R> {
    fn new(input_left: &OperationRef, input_right: &OperationRef, runner: R) -> OperationRef {
        let value = runner.run(
            input_left.as_ref().get_value(),
            input_right.as_ref().get_value(),
        );
        OperationRef::new(Self {
            input_left: input_left.clone(),
            input_right: input_right.clone(),

            value,
            runner,
        })
    }
}

impl<R: BinaryOperationRunner + 'static> Operation for BinaryOperation<R> {
    fn back_grad(&mut self, delta: Matrix) {
        let mut input_left_ref = self.input_left.as_mut();
        let mut input_right_ref = self.input_right.as_mut();

        let (delta_left, delta_right) = self.runner.grad(
            input_left_ref.get_value(),
            input_right_ref.get_value(),
            &delta,
        );

        input_left_ref.back_grad(delta_left);
        input_right_ref.back_grad(delta_right);
    }

    fn get_value(&self) -> &Matrix {
        &self.value
    }
}

/*------------------------------------------------------------------------------------------------*/
