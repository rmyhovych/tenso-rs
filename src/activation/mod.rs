use crate::operation::OperationRef;

pub mod sigmoid;


/*------------------------------------------------------------------------------------------------*/

pub trait Activation {
    fn run(&self, input: OperationRef) -> OperationRef;
}

/*------------------------------------------------------------------------------------------------*/

pub struct ActivationClosure {
    closure: Box<dyn Fn(OperationRef) -> OperationRef>,
}

impl ActivationClosure {
    pub fn new(closure: impl Fn(OperationRef) -> OperationRef + 'static) -> Self {
        Self {
            closure: Box::new(closure),
        }
    }
}

impl Activation for ActivationClosure {
    fn run(&self, input: OperationRef) -> OperationRef {
        self.closure.as_ref()(input)
    }
}
