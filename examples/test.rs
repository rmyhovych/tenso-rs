use tenso_rs::{
    matrix::Matrix,
    node::{constant::NodeConstant, variable::NodeVariable, NodeInternal},
    optim::{self, sgd::OptimFuncSGD, Optimizer},
};

fn main() {
    let x = NodeConstant::new(Matrix::new_value([2, 1], 1.0));
    let w = NodeVariable::new(Matrix::new_value([2, 2], 2.0));
    let b = NodeVariable::new(Matrix::new_value([2, 1], 3.0));

    let y_exp = NodeConstant::new(Matrix::new_value([2, 1], 5.0));

    let wx = w.matmul(&x);
    let y = wx.add(&b).sigmoid();

    let err = y.sub(&y_exp).pow(2.0).sum();
    print!("Err:\n{}\n", err);

    err.back();

    {
        let mut w_internal = w.get_internal();
        if let Some(var) = w_internal.try_get_variable() {
            print!("W:\n{}{}", var.get_value(), var.get_gradient());
        }
    }
    {
        let mut b_internal = b.get_internal();
        if let Some(var) = b_internal.try_get_variable() {
            print!("B:\n{}{}", var.get_value(), var.get_gradient());
        }
    }
}
