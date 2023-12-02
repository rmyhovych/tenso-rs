use dataset::{MnistDataset, MnistEntry};
use tenso_rs::{
    matrix::Matrix,
    model::{activation::{sigmoid::ActivationSigmoid, relu::ActivationRelu}, linear::ModelLinear, Model},
    node::{constant::NodeConstant, Node},
    optim::{sgd::OptimFuncSGD, Optimizer},
};

mod dataset;

struct ModelMnist {
    layers: Vec<ModelLinear>,
}

impl ModelMnist {
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
                ActivationRelu,
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

impl Model for ModelMnist {
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

fn main() {
    const CHUNK_SIZE: usize = 16;
    const TRAIN_SIZE: usize = 2048;

    let dataset = MnistDataset::load(
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte",
    );

    let mut xs: Vec<Node> = dataset
        .get_entries()
        .iter()
        .map(|e| {
            NodeConstant::new(Matrix::new_slice(
                [MnistEntry::IMAGE_WIDTH * MnistEntry::IMAGE_WIDTH, 1],
                &e.get_image().as_slice(),
            ))
        })
        .collect();
    xs.drain(TRAIN_SIZE..xs.len());

    let mut ys_exp: Vec<Node> = dataset
        .get_entries()
        .iter()
        .map(|e| NodeConstant::new(Matrix::new_one_hot(e.get_label(), 10).transpose()))
        .collect();
    ys_exp.drain(TRAIN_SIZE..ys_exp.len());

    let model = ModelMnist::new([MnistEntry::IMAGE_WIDTH * MnistEntry::IMAGE_WIDTH, 16, 10]);

    let mut optim = Optimizer::new(OptimFuncSGD::new(0.001));
    optim.add_model(&model);

    for i in 0..1000 {
        let mut error_mean = 0.0;
        for chunk_start in (0..xs.len()).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(xs.len());
            for c in chunk_start..chunk_end {
                let y = model.run(&xs[c]);
                let error = y.sub(&ys_exp[c]).pow(2.0).mean();
                error.back();

                error_mean += error.get_value()[[0, 0]];
            }

            optim.step();
        }

        println!("Error {}: {:.2}", i, error_mean / xs.len() as f32);
    }
}