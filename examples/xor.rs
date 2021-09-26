use tenso_rs::matrix::Matrix;
use tenso_rs::operation::{input::InputPlaceholder, Operation};
use tenso_rs::optim::RunningOptimizer;
use tenso_rs::optim::{sgd::SGDOptimizerRunner, Optimizer};

fn linear(
    input: &Operation,
    optim: &mut dyn Optimizer,
    in_size: usize,
    out_size: usize,
) -> Operation {
    let weights = Matrix::randn(out_size, in_size, 0.0, 1.0).as_variable(optim);
    let biases = Matrix::randn(out_size, 1, 0.0, 1.0).as_variable(optim);

    weights.cross(input.clone()) + biases
}

fn main() {
    let inputs: Vec<Matrix> = vec![
        Matrix::new(2, 1, vec![1.0, 0.0]),
        Matrix::new(2, 1, vec![0.0, 0.0]),
        Matrix::new(2, 1, vec![0.0, 1.0]),
        Matrix::new(2, 1, vec![1.0, 1.0]),
    ];
    let labels: Vec<Matrix> = vec![
        Matrix::new(1, 1, vec![1.0]),
        Matrix::new(1, 1, vec![0.0]),
        Matrix::new(1, 1, vec![1.0]),
        Matrix::new(1, 1, vec![0.0]),
    ];

    let mut input_ph = InputPlaceholder::new();
    let mut label_ph = InputPlaceholder::new();

    let mut optim = RunningOptimizer::new(SGDOptimizerRunner::new(0.1));
    let net = linear(
        &linear(&input_ph, &mut optim, 2, 5).sigmoid(),
        &mut optim,
        5,
        1,
    )
    .sigmoid();

    // net.add_to_optimizer(&mut optim);

    let mut loss_f = (label_ph.clone() - net.clone()).pow(2.0).sum();

    for _ in 0..200000 {
        let mut loss_sum: f32 = 0.0;
        for (input, label) in inputs.iter().zip(labels.iter()) {
            input_ph.set_input(input.clone());
            label_ph.set_input(label.clone());

            let loss = loss_f.run();
            loss_sum += loss[0][0];

            loss_f.back();
        }

        optim.step();
        println!("Loss: {}", loss_sum);
    }
}
