pub mod matrix;
pub mod operation;
pub mod optim;

use matrix::Matrix;
use optim::{Optimizer, sgd::SGDOptimizer};

fn main() {
    let mut matrix0 = Matrix::randn(5, 3, 0.0, 1.0);
    let mut matrix1 = Matrix::randn(5, 3, 0.0, 5.0);

    let mut optim = SGDOptimizer::new(0.001);
    let mut operation = matrix0.var_op(&mut optim).times(2.0).sum();

    println!("\n{}", operation.run());

    for _ in 0..20 {
        operation.back();
        optim.step();

        println!("\n{}", matrix0);
    }
}
