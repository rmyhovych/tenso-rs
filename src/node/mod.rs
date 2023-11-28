pub mod op;
pub mod variable;

use self::variable::NodeVariable;
use crate::matrix::Matrix;
use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
};

pub trait NodeInternal {
    fn get_value(&self) -> &Matrix;
    fn get_value_mut(&mut self) -> &mut Matrix;

    fn back_delta(&mut self, delta: Matrix);

    fn is_variable(&mut self) -> Option<&mut NodeVariable> {
        None
    }
}

pub struct NodeValueRef<'a> {
    node_ref: Ref<'a, dyn NodeInternal>,
}

impl<'a> NodeValueRef<'a> {
    fn new(node_ref: Ref<'a, dyn NodeInternal>) -> Self {
        Self { node_ref }
    }
}

impl<'a> Deref for NodeValueRef<'a> {
    type Target = Matrix;

    fn deref(&self) -> &Self::Target {
        self.node_ref.get_value()
    }
}

pub struct Node {
    node: Rc<RefCell<dyn NodeInternal>>,
}

impl Node {
    fn new<TNodeType: NodeInternal + 'static>(internal: TNodeType) -> Self {
        Self {
            node: Rc::new(RefCell::new(internal)),
        }
    }

    pub fn size(&self) -> [usize; 2] {
        self.node.borrow().get_value().size()
    }

    fn back(&mut self) {
        let delta = Matrix::new_value(self.size(), 1.0);
        self.node.borrow_mut().back_delta(delta);
    }

    pub fn get<'a>(&'a self) -> NodeValueRef<'a> {
        NodeValueRef::<'a>::new(self.node.borrow())
    }
}
