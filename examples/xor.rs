use tenso_rs::{
    matrix::Matrix,
    node::{constant::NodeConstant, variable::NodeVariable, Node},
    optim::{sgd::OptimFuncSGD, Optimizer},
};

fn get_input_matrices() -> Vec<Matrix> {
    let mut input_matrices = vec![
        Matrix::new_zero([2, 1]),
        Matrix::new_zero([2, 1]),
        Matrix::new_zero([2, 1]),
        Matrix::new_zero([2, 1]),
    ];

    for (i, mat) in input_matrices.iter_mut().enumerate() {
        mat[[0, 0]] = if i % 2 == 0 { 1.0 } else { 0.0 };
        mat[[1, 0]] = if i / 2 == 0 { 1.0 } else { 0.0 }
    }

    input_matrices
}

fn get_output_matrices(input: &Vec<Matrix>) -> Vec<Matrix> {
    let mut output_matrices = vec![
        Matrix::new_zero([1, 1]),
        Matrix::new_zero([1, 1]),
        Matrix::new_zero([1, 1]),
        Matrix::new_zero([1, 1]),
    ];

    output_matrices
        .iter_mut()
        .zip(input.iter())
        .for_each(|(v_out, v_in)| {
            let diff = (v_in[[0, 0]] - v_in[[1, 0]]).abs();
            v_out[[0, 0]] = if diff < 0.001 { 0.0 } else { 1.0 }
        });

    output_matrices
}

fn main() {
    let (xs, ys_exp) = {
        let input = get_input_matrices();
        let ys_exp = get_output_matrices(&input)
            .into_iter()
            .map(|m| NodeConstant::new(m))
            .collect::<Vec<Node>>();
        let xs = input
            .into_iter()
            .map(|m| NodeConstant::new(m))
            .collect::<Vec<Node>>();

        (xs, ys_exp)
    };

    let w0 = NodeVariable::new(Matrix::new_randn([2, 2], 0.0, 1.0));
    let b0: Node = NodeVariable::new(Matrix::new_randn([2, 1], 0.0, 1.0));

    let w1: Node = NodeVariable::new(Matrix::new_randn([1, 2], 0.0, 1.0));
    let b1: Node = NodeVariable::new(Matrix::new_randn([1, 1], 0.0, 1.0));

    let run_fn = |x: &Node| -> Node {
        let h0 = w0.matmul(x).add(&b0);
        let y = w1.matmul(&h0).add(&b1);

        y.sigmoid()
    };

    let mut optimizer = Optimizer::new(OptimFuncSGD::new(0.1));
    optimizer.add_variables(vec![w0.clone(), b0.clone(), w1.clone(), b1.clone()]);

    for i in 0..10000 {
        let mut full_error = NodeConstant::new(Matrix::new_zero([1, 1]));
        for (x, y_exp) in xs.iter().zip(ys_exp.iter()) {
            let y = run_fn(x);
            let error = y_exp.sub(&y).pow(2.0).sum();
            full_error = full_error.add(&error);
        }

        //println!("Error [{}]:\n{}", i, full_error);
        full_error.back();
        optimizer.step();
    }

    println!("------------------------------------------------------------------------");

    for (x, y_exp) in xs.iter().zip(ys_exp.iter()) {
        let y = run_fn(x);
        println!("X:\n{}Y\n{}YExp\n{}", x, y, y_exp);
    }
}
