use std::time::Instant;

use tenso_rs::matrix::Matrix;

pub fn main() {
    let size = (2, 7);

    let mut start = Instant::now();
    let mat = Matrix::randn(size, 0.0, 1.0);
    println!("Elapsed Init: {:?}", start.elapsed());

    start = Instant::now();
    let new_mat = mat.transpose();
    println!("Elapsed Transpose: {:?}", start.elapsed());

    start = Instant::now();
    let mult_mat = mat.matmul(&new_mat);
    println!("Elapsed MatMul: {:?}", start.elapsed());

    //println!("{}", mult_mat.get((0, 0)));

    println!("\n\nLeft:\n{}", mat);
    println!("Right (Transpose):\n{}", new_mat);

    println!("Mat Mul:\n{}", mult_mat);

    println!("Mat Mul Sum:\n{}", mult_mat.sum());
}
