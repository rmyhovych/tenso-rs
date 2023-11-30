use tenso_rs::matrix::Matrix;

pub fn main() {
    let matrix = Matrix::new_slice([2, 3], &[100.0, -2.0, 3.0, -1.0, 20.0, -3.0]);
    println!("{:.1}", matrix);
}
