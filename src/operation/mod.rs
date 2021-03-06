use crate::matrix::Matrix;
use std::{boxed::Box, cell::RefCell, rc::Rc};

pub mod basic;
pub mod input;
pub mod more;

/*------------------------------------------------------------------------------------------------*/

#[derive(Clone)]
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

    /*------------------------------------------------------*/

    fn back_grad(&mut self, grad: Matrix) {
        self.op.borrow_mut().back_grad(grad);
    }
}

/*------------------------------------------------------------------------------------------------*/

trait OperationBase {
    fn run(&mut self) -> Matrix;

    fn back(&mut self);

    fn back_grad(&mut self, grad: Matrix);

    fn get_output(&self) -> Matrix;

    fn set_input(&mut self, input: Matrix);
}

/*------------------------------------------------------------------------------------------------*/

struct UnaryOperation {
    op_input: Operation,

    func_output: Box<dyn Fn(&Matrix) -> Matrix + 'static>,
    func_grad: Box<dyn Fn(&mut Operation, &Matrix) -> () + 'static>,

    output: Matrix,
}

impl UnaryOperation {
    fn new(
        op_input: Operation,

        func_output: impl Fn(&Matrix) -> Matrix + 'static,
        func_grad: impl Fn(&mut Operation, &Matrix) -> () + 'static,
    ) -> Operation {
        Operation::new(Self {
            op_input,

            func_output: Box::new(func_output),
            func_grad: Box::new(func_grad),

            output: Matrix::zeros(0, 0),
        })
    }
}

impl OperationBase for UnaryOperation {
    fn run(&mut self) -> Matrix {
        let out_input = self.op_input.run();
        self.output = self.func_output.as_ref()(&out_input);

        self.output.clone()
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(self.output.get_height(), self.output.get_width(), 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.func_grad.as_ref()(&mut self.op_input, &grad);
    }

    fn get_output(&self) -> Matrix {
        self.output.clone()
    }

    fn set_input(&mut self, _: Matrix) {}
}

/*------------------------------------------------------------------------------------------------*/

struct BinaryOperation {
    op_left: Operation,
    op_right: Operation,

    func_output: Box<dyn Fn(&Matrix, &Matrix) -> Matrix + 'static>,
    func_grad: Box<dyn Fn(&mut Operation, &mut Operation, &Matrix) -> () + 'static>,

    output: Matrix,
}

impl BinaryOperation {
    fn new(
        op_left: Operation,
        op_right: Operation,

        func_output: impl Fn(&Matrix, &Matrix) -> Matrix + 'static,
        func_grad: impl Fn(&mut Operation, &mut Operation, &Matrix) -> () + 'static,
    ) -> Operation {
        Operation::new(Self {
            op_left: op_left,
            op_right: op_right,

            func_output: Box::new(func_output),
            func_grad: Box::new(func_grad),

            output: Matrix::zeros(0, 0),
        })
    }
}

impl OperationBase for BinaryOperation {
    fn run(&mut self) -> Matrix {
        let out_left = self.op_left.run();
        let out_right = self.op_right.run();

        self.output = self.func_output.as_ref()(&out_left, &out_right);

        self.output.clone()
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(self.output.get_height(), self.output.get_width(), 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.func_grad.as_ref()(&mut self.op_left, &mut self.op_right, &grad);
    }

    fn get_output(&self) -> Matrix {
        self.output.clone()
    }

    fn set_input(&mut self, _: Matrix) {}
}

/*------------------------------------------------------------------------------------------------*/
