use std::{cell::RefCell, rc::Rc, ops::{Index, IndexMut}};

use crate::matrix::Matrix;

pub trait NodeInternal {
    fn get_value(&self) -> &Matrix;
    fn get_value_mut(&mut self) -> &mut Matrix;

    fn is_variable(&mut self) -> Option<&mut NodeVariable> {
        None
    }
}

pub struct NodeVariable {
    value: Matrix,
    gradient: Matrix,
}

impl NodeInternal for NodeVariable {
    fn is_variable(&mut self) -> Option<&mut NodeVariable> {
        Some(self)
    }

    fn get_value(&self) -> &Matrix {
        &self.value
    }

    fn get_value_mut(&mut self) -> &mut Matrix {
        &mut self.value
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
}

impl Index<[usize; 2]> for Node {
    type Output = f32;

    
}