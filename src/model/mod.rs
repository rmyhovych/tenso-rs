pub mod linear;
pub mod activation;

use crate::node::Node;

pub trait Model {
    fn run(&self, x: &Node) -> Node;

    fn for_each_variable<TFuncType: FnMut(&Node)>(&self, func: &mut TFuncType);
}
