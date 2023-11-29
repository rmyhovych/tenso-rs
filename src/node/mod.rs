pub mod constant;
pub mod op;
pub mod variable;

use self::variable::NodeVariable;
use crate::matrix::Matrix;
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Display,
    ops::{Deref, DerefMut},
    rc::Rc,
};

pub trait NodeInternal {
    fn get_value(&self) -> &Matrix;

    fn back_delta(&mut self, delta: Matrix);

    fn try_get_variable(&mut self) -> Option<&mut NodeVariable> {
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

pub struct NodeInternalRef<'a> {
    node_ref: RefMut<'a, dyn NodeInternal>,
}

impl<'a> NodeInternalRef<'a> {
    fn new(node_ref: RefMut<'a, dyn NodeInternal>) -> Self {
        Self { node_ref }
    }
}

impl<'a> Deref for NodeInternalRef<'a> {
    type Target = dyn NodeInternal + 'a;

    fn deref(&self) -> &Self::Target {
        self.node_ref.deref()
    }
}

impl<'a> DerefMut for NodeInternalRef<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.node_ref.deref_mut()
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

    pub fn back(&self) {
        let delta = Matrix::new_value(self.size(), 1.0);
        self.back_delta(delta);
    }

    pub fn is_variable(&self) -> bool {
        self.node.borrow_mut().try_get_variable().is_some()
    }

    pub fn get_value<'a>(&'a self) -> NodeValueRef<'a> {
        NodeValueRef::<'a>::new(self.node.borrow())
    }

    pub fn get_internal<'a>(&'a self) -> NodeInternalRef<'a> {
        NodeInternalRef::<'a>::new(self.node.borrow_mut())
    }

    fn back_delta(&self, delta: Matrix) {
        self.node.borrow_mut().back_delta(delta);
    }
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get_value().fmt(f)
    }
}
