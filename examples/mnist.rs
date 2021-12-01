use std::{
    error::Error,
    fs::{metadata, File},
    io::Read,
};

use plotters::prelude::*;
use rand::{seq::index::sample, thread_rng};
use tenso_rs::operation::{input::InputPlaceholder, Operation};
use tenso_rs::optim::{sgd::SGDOptimizerRunner, Optimizer};
use tenso_rs::{matrix::Matrix, optim::RunningOptimizer};

fn linear(input: &Operation, in_size: usize, out_size: usize) -> Operation {
    let weights = Matrix::randn(out_size, in_size, 0.0, 1.0).as_variable();
    let biases = Matrix::randn(out_size, 1, 0.0, 1.0).as_variable();

    weights.mmul(input.clone()) + biases
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
    debug_assert_eq!(raw_data.len() % len, 0);

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

fn plot_data(losses: Vec<f32>, accuracies: Vec<f32>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("examples/mnist_result.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE);
    let root = root.margin(10, 10, 10, 10);

    let mut chart = ChartBuilder::on(&root)
        // Set the caption of the chart
        .caption("This is our first plot", ("sans-serif", 40).into_font())
        // Set the size of the label region
        .x_label_area_size(20)
        .y_label_area_size(40)
        // Finally attach a coordinate on the drawing area and make a chart context
        .build_cartesian_2d(
            0f32..(losses.len().max(accuracies.len()) as f32),
            0f32..1f32,
        )?;

    // Then we can draw a mesh
    chart
        .configure_mesh()
        // We can customize the maximum number of labels allowed for each axis
        .x_labels(5)
        .y_labels(5)
        // We can also change the format of the label text
        .y_label_formatter(&|x| format!("{:.3}", x))
        .draw()?;

    chart.draw_series(LineSeries::new(
        (0..losses.len())
            .zip(losses.into_iter())
            .into_iter()
            .map(|(index, loss)| (index as f32, loss)),
        &RED,
    ))?;

    chart.draw_series(LineSeries::new(
        (0..accuracies.len())
            .zip(accuracies.into_iter())
            .into_iter()
            .map(|(index, accuracy)| (index as f32, accuracy)),
        &BLUE,
    ))?;

    Ok(())
}

fn get_rolling_average(data: &Vec<f32>, size: usize) -> f32 {
    let slice_start = (data.len() as i32 - size as i32).max(0) as usize;
    data.as_slice()[slice_start..].iter().sum::<f32>() / (size.min(data.len()) as f32)
}

fn main() {
    const SAMPLE_SIZE: usize = 16;

    let mut image_data = read_binary_file("data/mnist/train-images-idx3-ubyte");
    image_data.drain(0..16);

    let mut label_data = read_binary_file("data/mnist/train-labels-idx1-ubyte");
    label_data.drain(0..8);

    let in_size: usize = 28 * 28;
    let out_size: usize = 10;

    let inputs: Vec<Matrix> = to_vectors(image_data, in_size);
    let labels: Vec<Matrix> = to_one_hot_vectors(label_data, out_size);

    let mut input_ph = InputPlaceholder::new();
    let mut label_ph = InputPlaceholder::new();

    let mut net = linear(&input_ph, in_size, 16).sigmoid();
    net = linear(&net, 16, out_size).sigmoid();

    let mut optim = RunningOptimizer::new(SGDOptimizerRunner::new(0.002));
    net.add_to_optimizer(&mut optim);

    let mut loss_f = (label_ph.clone() - net.clone()).pow(2.0).sum();

    let mut losses: Vec<f32> = Vec::new();
    let mut accuracies: Vec<f32> = Vec::new();

    let mut average_losses: Vec<f32> = Vec::new();
    let mut average_accuracies: Vec<f32> = Vec::new();


    const DISPLAY_ROLLING_AVERAGE: usize = 1000;

    let mut rng = thread_rng();
    for i in 0..200000 {
        let mut loss_sum: f32 = 0.0;

        let mut n_accurate: u32 = 0;
        for (input, label) in sample(&mut rng, inputs.len(), SAMPLE_SIZE)
            .into_iter()
            .map(|index| (&inputs[index], &labels[index]))
        {
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

        losses.push(loss_sum / SAMPLE_SIZE as f32);
        accuracies.push(n_accurate as f32 / SAMPLE_SIZE as f32);

        if i % DISPLAY_ROLLING_AVERAGE == 0 {
            let average_accuracy = get_rolling_average(&accuracies, DISPLAY_ROLLING_AVERAGE);
            let average_loss = get_rolling_average(&losses, DISPLAY_ROLLING_AVERAGE);

            average_accuracies.push(average_accuracy);
            average_losses.push(average_loss);

            println!("Accuracy: {}", average_accuracy);
            println!("Loss: {}\n", average_loss);
        }
    }

    plot_data(average_losses, average_accuracies).unwrap();
}
