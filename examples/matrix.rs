use std::time::Instant;

use tenso_rs::matrix::Matrix;

pub fn main() {
    let size = [1, 2];

    let mut start = Instant::now();
    let mat = Matrix::new_randn(size, 0.0, 1.0);
    println!("Elapsed Init: {:?}", start.elapsed());
    println!("Left:\n{}", mat);

    start = Instant::now();
    let new_mat = mat.transpose();
    println!("Elapsed Transpose: {:?}", start.elapsed());
    println!("Right (Transpose):\n{}", new_mat);

    start = Instant::now();
    let mult_mat = mat.matmul(&new_mat);
    println!("Elapsed MatMul: {:?}", start.elapsed());
    println!("Mat Mul:\n{}", mult_mat);

    start = Instant::now();
    let sum_mat = mult_mat.sum();
    println!("Elapsed Sum: {:?}", start.elapsed());

    println!("Mat Mul Sum:\n{}", sum_mat);
}
