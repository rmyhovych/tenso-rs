use crate::matrix::Matrix;

pub mod sgd;

struct OptimVariables {
    variables: Vec<(Matrix, Matrix)>,
}

impl OptimVariables {
    fn new() -> Self {
        Self {
            variables: Vec::new(),
        }
    }

    fn add(&mut self, value: Matrix, grad: Matrix) {
        self.variables.push((value, grad));
    }
}

pub trait Optimizer {
    fn add_var(&mut self, value: Matrix, grad: Matrix);
    
    fn step(&mut self);
}
