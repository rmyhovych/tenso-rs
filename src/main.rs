pub mod matrix;
pub mod operation;

use matrix::Matrix;
use operation::Operation;

fn main() {
    let m1 = Matrix::randn(3, 2, 0.0, 1.0);

    let mut op = m1.var();

    println!("\n{}", op.run());
}
