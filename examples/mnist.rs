use std::{
    fs::{metadata, File},
    io::Read,
};

use tenso_rs::operation::{input::InputPlaceholder, Operation};
use tenso_rs::optim::{sgd::SGDOptimizerRunner, Optimizer};
use tenso_rs::{matrix::Matrix, optim::RunningOptimizer};

fn linear(input: &Operation, in_size: usize, out_size: usize) -> Operation {
    let weights = Matrix::randn(out_size, in_size, 0.0, 1.0).as_variable();
    let biases = Matrix::randn(out_size, 1, 0.0, 1.0).as_variable();

    weights.cross(input.clone()) + biases
}

fn read_binary_file(filename: &str) -> Vec<u8> {
    match File::open(filename) {
        Ok(mut file) => {
            let file_len = match metadata(filename) {
                Ok(meta) => meta.len(),
                Err(_) => 0,
            };

            let mut buffer: Vec<u8> = vec![0; file_len as usize];
            file.read(buffer.as_mut_slice()).expect("Buffer Overflow");

            buffer
        }
        Err(_) => Vec::new(),
    }
}

fn to_vectors(raw_data: Vec<u8>, len: usize) -> Vec<Matrix> {
    assert_eq!(raw_data.len() % len, 0);

    let mut matrices = Vec::<Matrix>::with_capacity(raw_data.len() / len);

    for chunk in raw_data.chunks(len) {
        let mat = Matrix::new(len, 1, chunk.iter().map(|v| *v as f32).collect());
        matrices.push(mat);
    }

    matrices
}

fn to_one_hot_vectors(raw_data: Vec<u8>, len: usize) -> Vec<Matrix> {
    let mut matrices = Vec::<Matrix>::with_capacity(raw_data.len() / len);

    for value in raw_data {
        let mut one_hot: Vec<f32> = vec![0.0; len];
        one_hot[value as usize] = 1.0;

        let mat = Matrix::new(len, 1, one_hot);
        matrices.push(mat);
    }

    matrices
}

fn is_accurate(y: &Matrix, label: &Matrix) -> bool {
    let mut max_index: usize = 0;
    let mut max_val: f32 = 0.0;
    let mut max_real_index = 0;

    for i in 0..10 {
        let label_val = label[i][0];
        if label_val == 1.0 {
            max_real_index = i;
        }

        let val = y[i][0];
        if val > max_val {
            max_index = i;
            max_val = val;
        }
    }

    max_index == max_real_index
}

fn main() {
    let mut image_data = read_binary_file("data/mnist/train-images-idx3-ubyte");
    image_data.drain(0..16);

    let mut label_data = read_binary_file("data/mnist/train-labels-idx1-ubyte");
    label_data.drain(0..8);

    let in_size: usize = 28 * 28;
    let out_size: usize = 10;

    let mut inputs: Vec<Matrix> = to_vectors(image_data, in_size);
    inputs.truncate(5);

    let mut labels: Vec<Matrix> = to_one_hot_vectors(label_data, out_size);
    labels.truncate(5);

    let mut input_ph = InputPlaceholder::new();
    let mut label_ph = InputPlaceholder::new();

    let mut net = linear(&input_ph, in_size, 16).sigmoid();
    net = linear(&net, 16, out_size).sigmoid();

    let mut optim = RunningOptimizer::new(SGDOptimizerRunner::new(0.01));
    net.add_to_optimizer(&mut optim);

    let mut loss_f = (label_ph.clone() - net.clone()).pow(2.0).sum();

    for _ in 0..2000 {
        let mut loss_sum: f32 = 0.0;

        let mut n_accurate: u32 = 0;
        for (input, label) in inputs.iter().zip(labels.iter()) {
            input_ph.set_input(input.clone());
            label_ph.set_input(label.clone());

            let loss = loss_f.run();
            loss_sum += loss[0][0];

            let net_out = net.get_output();
            if is_accurate(&net_out, label) {
                n_accurate += 1;
            }

            loss_f.back();
        }

        optim.step();
        println!("Accuracy: {}", n_accurate as f32 / inputs.len() as f32);
        println!("Loss: {}\n", loss_sum / inputs.len() as f32);
    }
}
