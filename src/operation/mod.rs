use crate::matrix::Matrix;
use std::{
    boxed::Box,
    cell::{Ref, RefCell, RefMut},
    ops::{Deref, DerefMut},
    rc::Rc,
};

pub mod basic;

/*------------------------------------------------------------------------------------------------*/

const EMPTY_GRAD: Matrix = Matrix::zeros(0, 0);

pub trait Operation {
    fn run(&mut self) -> &Matrix;

    fn back(&mut self);

    fn back_grad(&mut self, grad: Matrix);

    fn get_output(&self) -> &Matrix;

    fn get_grad(&self) -> &Matrix;
}

pub struct OperationRef {
    operation: Rc<RefCell<dyn Operation>>,
}

impl Operation for OperationRef {
    fn run(&mut self) -> &Matrix {
        self.get_mut().run()
    }

    fn back(&mut self) {
        self.get_mut().back();
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.get_mut().back_grad(grad);
    }

    fn get_output(&self) -> &Matrix {
        self.get().get_output()
    }

    fn get_grad(&self) -> &Matrix {
        self.get().get_grad()
    }
}

impl OperationRef {
    pub fn new<T>(base_op: T) -> Self
    where
        T: Operation + 'static,
    {
        Self {
            operation: Rc::new(RefCell::new(base_op)),
        }
    }

    pub fn get(&self) -> Ref<'_, dyn Operation> {
        self.operation.as_ref().borrow()
    }

    pub fn get_mut(&self) -> RefMut<'_, dyn Operation> {
        self.operation.as_ref().borrow_mut()
    }
}

impl Clone for OperationRef {
    fn clone(&self) -> Self {
        Self {
            operation: self.operation.clone(),
        }
    }
}

/*------------------------------------------------------------------------------------------------*/

struct Variable {
    value: Matrix,
    grad: Matrix,
}

impl Variable {
    fn new(value: Matrix) -> OperationRef {
        let grad = Matrix::zeros(value.height, value.width);
        OperationRef::new(Self { value, grad })
    }
}

impl Operation for Variable {
    fn run(&mut self) -> &Matrix {
        self.get_output()
    }

    fn back(&mut self) {}

    fn back_grad(&mut self, grad: Matrix) {
        self.grad = grad
    }

    fn get_output(&self) -> &Matrix {
        &self.value
    }

    fn get_grad(&self) -> &Matrix {
        &self.grad
    }
}

impl Matrix {
    pub fn var(self) -> OperationRef {
        Variable::new(self)
    }
}

/*------------------------------------------------------------------------------------------------*/

struct UnaryOperation {
    op_input: OperationRef,

    func_output: Box<dyn Fn(&Matrix) -> Matrix + 'static>,
    func_grad: Box<dyn Fn(&mut dyn Operation, &Matrix) -> () + 'static>,

    output: Matrix,
}

impl UnaryOperation {
    fn new(
        op_input: OperationRef,

        func_output: impl Fn(&Matrix) -> Matrix + 'static,
        func_grad: impl Fn(&mut dyn Operation, &Matrix) -> () + 'static,
    ) -> OperationRef {
        OperationRef::new(Self {
            op_input,

            func_output: Box::new(func_output),
            func_grad: Box::new(func_grad),

            output: Matrix::zeros(0, 0),
        })
    }
}

impl Operation for UnaryOperation {
    fn run(&mut self) -> &Matrix {
        let out_input = self.op_input.get_mut().run();
        self.output = self.func_output.as_ref()(out_input);

        &self.output
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(self.output.height, self.output.width, 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.func_grad.as_ref()(self.op_input.get_mut().deref_mut(), &grad);
    }

    fn get_output(&self) -> &Matrix {
        &self.output
    }

    fn get_grad(&self) -> &Matrix {
        &EMPTY_GRAD
    }
}

/*------------------------------------------------------------------------------------------------*/

struct BinaryOperation {
    op_left: OperationRef,
    op_right: OperationRef,

    func_output: Box<dyn Fn(&Matrix, &Matrix) -> Matrix + 'static>,
    func_grad: Box<dyn Fn(&mut dyn Operation, &mut dyn Operation, &Matrix) -> () + 'static>,

    output: Matrix,
}

impl BinaryOperation {
    fn new(
        op_left: OperationRef,
        op_right: OperationRef,

        func_output: impl Fn(&Matrix, &Matrix) -> Matrix + 'static,
        func_grad: impl Fn(&mut dyn Operation, &mut dyn Operation, &Matrix) -> () + 'static,
    ) -> OperationRef {
        OperationRef::new(Self {
            op_left,
            op_right,

            func_output: Box::new(func_output),
            func_grad: Box::new(func_grad),

            output: Matrix::zeros(0, 0),
        })
    }
}

impl Operation for BinaryOperation {
    fn run(&mut self) -> &Matrix {
        let out_left = self.op_left.get_mut().run();
        let out_right = self.op_right.get_mut().run();

        self.output = self.func_output.as_ref()(out_left, out_right);

        &self.output
    }

    fn back(&mut self) {
        let grad = Matrix::from_const(self.output.height, self.output.width, 1.0);
        self.back_grad(grad);
    }

    fn back_grad(&mut self, grad: Matrix) {
        self.func_grad.as_ref()(
            self.op_left.get_mut().deref_mut(),
            self.op_right.get_mut().deref_mut(),
            &grad,
        );
    }

    fn get_output(&self) -> &Matrix {
        &self.output
    }

    fn get_grad(&self) -> &Matrix {
        &EMPTY_GRAD
    }
}

/*------------------------------------------------------------------------------------------------*/
