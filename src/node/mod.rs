use std::{cell::RefCell, rc::Rc};

pub trait NodeInternal {
    fn is_variable(&mut self) -> Option<&mut NodeVariable> {
        None
    }
}

pub struct NodeVariable {}

impl NodeInternal for NodeVariable {
    fn is_variable(&mut self) -> Option<&mut NodeVariable> {
        Some(self)
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
}
