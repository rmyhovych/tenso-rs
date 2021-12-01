use std::marker::PhantomData;

use crate::{
    matrix::Matrix,
    operation::{OperationRef, Variable},
};

pub mod feedforward;
pub mod linear;

/*------------------------------------------------------------------------------------------------*/

pub struct VariableGroup {
    variables: Vec<OperationRef>,
}

impl VariableGroup {
    fn new() -> Self {
        Self {
            variables: Vec::new(),
        }
    }

    pub fn add(&mut self, initial_value: Matrix) -> OperationRef {
        self.variables.push(Variable::new(initial_value));
        self.variables.last().unwrap().clone()
    }
}

/*------------------------------------------------------------------------------------------------*/

pub trait ModuleRunner<D> {
    fn new(data: D, variables: &mut VariableGroup) -> Self;

    fn run(&self, input: OperationRef) -> OperationRef;
}

pub struct ModuleBase<D, R: ModuleRunner<D> + 'static> {
    variable_group: VariableGroup,

    runner: R,

    __phantom: PhantomData<D>,
}

pub trait Module {
    fn get_variables(&self) -> &Vec<OperationRef>;

    fn run(&self, input: &OperationRef) -> OperationRef;
}

impl<D, R: ModuleRunner<D> + 'static> ModuleBase<D, R> {
    pub fn new(data: D) -> Self {
        let mut variable_group = VariableGroup::new();
        let runner = ModuleRunner::new(data, &mut variable_group);

        Self {
            variable_group,
            runner,

            __phantom: PhantomData,
        }
    }
}

impl<D, R: ModuleRunner<D> + 'static> Module for ModuleBase<D, R> {
    fn run(&self, input: &OperationRef) -> OperationRef {
        self.runner.run(input.clone())
    }

    fn get_variables(&self) -> &Vec<OperationRef> {
        &self.variable_group.variables
    }
}
