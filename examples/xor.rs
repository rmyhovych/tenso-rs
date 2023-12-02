use std::fmt::Display;

use tenso_rs::{
    matrix::Matrix,
    model::{
        activation::{leaky_relu::ActivationLeakyRelu, sigmoid::ActivationSigmoid},
        linear::ModelLinear,
        Model,
    },
    node::{constant::NodeConstant, Node},
    optim::{sgd::OptimFuncSGD, Optimizer},
};

fn get_input_matrices() -> Vec<Matrix> {
    vec![
        Matrix::new_slice([2, 1], &[0.0, 0.0]),
        Matrix::new_slice([2, 1], &[1.0, 0.0]),
        Matrix::new_slice([2, 1], &[0.0, 1.0]),
        Matrix::new_slice([2, 1], &[1.0, 1.0]),
    ]
}

fn get_output_matrices() -> Vec<Matrix> {
    vec![
        Matrix::new_slice([1, 1], &[0.0]),
        Matrix::new_slice([1, 1], &[1.0]),
        Matrix::new_slice([1, 1], &[1.0]),
        Matrix::new_slice([1, 1], &[0.0]),
    ]
}

struct ModelXOR {
    layers: Vec<ModelLinear>,
}

impl ModelXOR {
    fn new<const LAYER_COUNT: usize>(layer_sizes: [usize; LAYER_COUNT]) -> Self {
        assert!(layer_sizes.len() >= 2);

        let layer_count = layer_sizes.len();
        let mut layers = Vec::with_capacity(layer_count - 1);
        let mut size_in = layer_sizes[0];
        for i in 1..(layer_count - 1) {
            let size_out = layer_sizes[i];
            layers.push(ModelLinear::new_activated(
                size_in,
                size_out,
                ActivationLeakyRelu::new(0.1),
            ));
            size_in = size_out;
        }

        let size_out = layer_sizes[layer_count - 1];
        layers.push(ModelLinear::new_activated(
            size_in,
            size_out,
            ActivationSigmoid,
        ));

        Self { layers }
    }
}

impl Model for ModelXOR {
    fn run(&self, x: &Node) -> Node {
        let mut y = x.clone();
        for layer in &self.layers {
            y = layer.run(&y);
        }

        y
    }

    fn for_each_variable<TFuncType: FnMut(&Node)>(&self, func: &mut TFuncType) {
        for layer in &self.layers {
            layer.for_each_variable(func);
        }
    }
}

impl Display for ModelXOR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, layer) in self.layers.iter().enumerate() {
            f.write_fmt(format_args!("\tLayer {}:\n", i))?;
            layer.fmt(f)?;
        }

        Ok(())
    }
}

fn main() {
    let (xs, ys_exp) = {
        let xs = get_input_matrices()
            .into_iter()
            .map(|m| NodeConstant::new(m))
            .collect::<Vec<Node>>();

        let ys_exp = get_output_matrices()
            .into_iter()
            .map(|m| NodeConstant::new(m))
            .collect::<Vec<Node>>();

        (xs, ys_exp)
    };

    let model = ModelXOR::new([2, 16, 16, 1]);

    let mut optimizer = Optimizer::new(OptimFuncSGD::new(0.01));
    optimizer.add_model(&model);

    let mut last_error = 0.0;
    for _ in 0..50000 {
        last_error = 0.0;
        for (x, y_exp) in xs.iter().zip(ys_exp.iter()) {
            let y: Node = model.run(x);
            let diff = y_exp.sub(&y);
            let error = diff.pow(2.0).mean();
            error.back();

            last_error += error.get_value()[[0, 0]];
        }

        optimizer.step();
    }

    //println!("{}", model);

    //println!("------------------------------------------------------------------------");
    for (x, y_exp) in xs.iter().zip(ys_exp.iter()) {
        let y = model.run(x);
        println!("X:\n{}Y\n{}YExp\n{}", x, y, y_exp);
    }

    println!("Error: {:.3}\n", last_error);
}
