pub mod leaky_relu;
pub mod relu;
pub mod sigmoid;

use crate::node::Node;

pub trait Activation {
    fn run(&self, node: &Node) -> Node;
}

pub struct EmptyActivation;
impl Activation for EmptyActivation {
    fn run(&self, node: &Node) -> Node {
        node.clone()
    }
}
