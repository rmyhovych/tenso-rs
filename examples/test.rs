use tenso_rs::{
    matrix::Matrix,
    node::{constant::NodeConstant, variable::NodeVariable},
};

fn main() {
    let x = NodeConstant::new(Matrix::new_slice([2, 2], &[0.1, -0.2, 0.1, -0.2]));
    let w = NodeVariable::new(Matrix::new_slice([2, 2], &[0.4, -0.3, -0.2, 0.1]));
    let b = NodeVariable::new(Matrix::new_slice([2, 2], &[-0.5, 0.16, -0.5, 0.16]));

    let y_exp = NodeConstant::new(Matrix::new_value([2, 2], 5.0));

    let wx = w.matmul(&x);
    let y = wx.add(&b).sigmoid();

    let err = y.sub(&y_exp).pow(2.0).sum();
    print!("Err:\n{}\n", err);

    err.back();

    {
        let mut w_internal = w.get_internal();
        if let Some(var) = w_internal.try_get_variable() {
            var.access(&mut |val, grad| {
                print!("W:\n{:.4}{:.4}", val, grad);
            })
        }
    }
    {
        let mut b_internal = b.get_internal();
        if let Some(var) = b_internal.try_get_variable() {
            var.access(&mut |val, grad| {
                print!("B:\n{:.4}{:.4}", val, grad);
            })
        }
    }
}
