use crate::operation::OperationRef;

use super::Activation;

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn run(&self, input: OperationRef) -> OperationRef {
        input.sigmoid()
    }
}
