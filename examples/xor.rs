use tenso_rs::activation::ActivationClosure;
use tenso_rs::matrix::Matrix;
use tenso_rs::module::feedforward::{
    FeedforwardLayers, FeedforwardModule, FeedforwardModuleRunner,
};
use tenso_rs::module::{Module, ModuleBase};
use tenso_rs::operation::{Operation, OperationRef};
use tenso_rs::optim::sgd::SGDOptimizerRunner;
use tenso_rs::optim::OptimizerBase;

fn main() {
    let inputs: Vec<OperationRef> = vec![
        Matrix::new(2, 1, vec![1.0, 0.0]),
        Matrix::new(2, 1, vec![0.0, 0.0]),
        Matrix::new(2, 1, vec![0.0, 1.0]),
        Matrix::new(2, 1, vec![1.0, 1.0]),
    ]
    .into_iter()
    .map(|m| m.to_const())
    .collect();

    let labels: Vec<OperationRef> = vec![
        Matrix::new(1, 1, vec![1.0]),
        Matrix::new(1, 1, vec![0.0]),
        Matrix::new(1, 1, vec![1.0]),
        Matrix::new(1, 1, vec![0.0]),
    ]
    .into_iter()
    .map(|m| m.to_const())
    .collect();

    let net = FeedforwardModule::new(
        FeedforwardLayers::new(2)
            .push(5, ActivationClosure::new(|op| op.sigmoid()))
            .push(1, ActivationClosure::new(|op| op.sigmoid())),
    );

    let mut optim = OptimizerBase::new(SGDOptimizerRunner::new(0.01), &net);

    let loss_f = |y: OperationRef, label: OperationRef| (y - label).pow(2.0).sum();

    for _ in 0..200000 {
        let mut loss_sum: f32 = 0.0;
        for (input, label) in inputs.iter().zip(labels.iter()) {
            let y = net.run(input);
            let mut loss = loss_f(y.clone(), label.clone());
            loss_sum += loss.as_ref().get_value()[0][0];

            if loss_sum < 0.001 {
                println!(
                    "y[{}], label[{}]",
                    y.as_ref().get_value()[0][0],
                    label.as_ref().get_value()[0][0]
                );
            }

            loss.back();
        }

        optim.step();
        println!("Loss: {}\n\n", loss_sum);
    }
}
