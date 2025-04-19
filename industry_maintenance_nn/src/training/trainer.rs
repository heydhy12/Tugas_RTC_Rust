use ndarray::{Array2, Axis};
use plotters::style::full_palette::BROWN;
use crate::model::neural_net::NeuralNetwork;
use crate::model::activations::{relu, relu_derivative};
use crate::utils::metrics::accuracy;  
use serde::Serialize;
use plotters::prelude::*;
use std::time::Instant; 


#[derive(Serialize, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy: f64, 
    pub test_loss: Option<f64>,
    pub test_accuracy: Option<f64>,
}

pub struct TrainingHistory {
    pub epochs: Vec<f64>,
    pub train_losses: Vec<f64>,
    pub train_accuracies: Vec<f64>, 
    pub test_losses: Vec<f64>,
    pub test_accuracies: Vec<f64>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        TrainingHistory {
            epochs: Vec::new(),
            train_losses: Vec::new(),
            train_accuracies: Vec::new(),
            test_losses: Vec::new(),
            test_accuracies: Vec::new(),
        }
    }

    pub fn add_metrics(&mut self, metrics: &TrainingMetrics) {
        self.epochs.push(metrics.epoch as f64);
        self.train_losses.push(metrics.train_loss);
        self.train_accuracies.push(metrics.train_accuracy);
        
        if let Some(test_loss) = metrics.test_loss {
            self.test_losses.push(test_loss);
        }
        
        if let Some(test_acc) = metrics.test_accuracy {
            self.test_accuracies.push(test_acc);
        }
    }

    pub fn plot(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(file_path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_epoch = *self.epochs.last().unwrap_or(&1.0);
        let max_loss = self.train_losses.iter().chain(self.test_losses.iter())
            .fold(0.0_f64, |a, &b| a.max(b)) * 1.1;

        let mut chart = ChartBuilder::on(&root)
            .caption("Training Progress", ("sans-serif", 40).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .right_y_label_area_size(40)
            .build_cartesian_2d(0f64..max_epoch, 0f64..max_loss)?
            .set_secondary_coord(0f64..max_epoch, 0f64..1.0);

        chart.configure_mesh()
            .x_desc("Epoch")
            .y_desc("Loss")
            .draw()?;

        chart.configure_secondary_axes()
            .y_desc("Accuracy")
            .draw()?;

        // Draw training loss line (blue)
        chart.draw_series(LineSeries::new(
            self.epochs.iter().zip(self.train_losses.iter()).map(|(x, y)| (*x, *y)),
            &BLUE,
        ))?.label("Training Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        // Draw test loss line if available (red)
        if !self.test_losses.is_empty() {
            chart.draw_series(LineSeries::new(
                self.epochs.iter().zip(self.test_losses.iter()).map(|(x, y)| (*x, *y)),
                &RED,
            ))?.label("Test Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        }

        // Draw training accuracy line (cyan)
        chart.draw_secondary_series(LineSeries::new(
            self.epochs.iter().zip(self.train_accuracies.iter()).map(|(x, y)| (*x, *y)),
            &BROWN,
        ))?.label("Training Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BROWN));

        // Draw test accuracy line if available (green)
        if !self.test_accuracies.is_empty() {
            chart.draw_secondary_series(LineSeries::new(
                self.epochs.iter().zip(self.test_accuracies.iter()).map(|(x, y)| (*x, *y)),
                &GREEN,
            ))?.label("Test Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        }

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }
}

pub fn train(
    model: &mut NeuralNetwork,
    x_train: &Array2<f64>,
    y_train: &Array2<f64>,
    x_test: Option<&Array2<f64>>,
    y_test: Option<&Array2<f64>>,
    learning_rate: f64,
    epochs: usize,
    print_every: usize,
) -> Vec<TrainingMetrics> {
    let start_time = Instant::now();  // Start timer
    let mut metrics_history = Vec::new();
    let mut training_history = TrainingHistory::new();

    println!("Training started...\n");

    for epoch in 0..epochs {
        // Forward pass
        let hidden_input1 = x_train.dot(&model.weights1) + &model.bias1;
        let hidden_output1 = relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&model.weights2) + &model.bias2;
        let hidden_output2 = relu(&hidden_input2);

        let output_input = hidden_output2.dot(&model.weights3) + &model.bias3;
        let output = relu(&output_input);

        // Backward pass
        let output_error = &output - y_train;
        let hidden_error2 = output_error.dot(&model.weights3.t()) * relu_derivative(&hidden_output2);
        let hidden_error1 = hidden_error2.dot(&model.weights2.t()) * relu_derivative(&hidden_output1);

        // Update weights and biases
        model.weights3 -= &(learning_rate * hidden_output2.t().dot(&output_error));
        model.bias3 -= &(learning_rate * output_error.sum_axis(Axis(0)));

        model.weights2 -= &(learning_rate * hidden_output1.t().dot(&hidden_error2));
        model.bias2 -= &(learning_rate * hidden_error2.sum_axis(Axis(0)));

        model.weights1 -= &(learning_rate * x_train.t().dot(&hidden_error1));
        model.bias1 -= &(learning_rate * hidden_error1.sum_axis(Axis(0)));

        // Calculate and store metrics
        if epoch % print_every == 0 || epoch == epochs - 1 {
            let train_loss = cross_entropy_loss(&output, y_train);
            let train_accuracy = accuracy(&output, y_train);
            
            let (test_loss, test_accuracy) = if let (Some(x_test), Some(y_test)) = (x_test, y_test) {
                let test_output = predict(model, x_test);
                (
                    Some(cross_entropy_loss(&test_output, y_test)),
                    Some(accuracy(&test_output, y_test))
                )
            } else {
                (None, None)
            };

            // Create TrainingMetrics properly with all fields
            let current_metrics = TrainingMetrics {
                epoch,
                train_loss,
                train_accuracy, 
                test_loss,
                test_accuracy,
                };
    
                metrics_history.push(current_metrics.clone());
                training_history.add_metrics(&current_metrics);

            if epoch % 100 == 0 {
                println!("Epoch: {}, Loss: {}", epoch, train_loss);
            }

            print_progress(epoch, epochs, &current_metrics);
        }
    }
    
    let duration = start_time.elapsed();
    println!("\nTraining completed in: {:.2?}", duration);

    // Save model and plot after training
    if let Err(e) = model.save("model_trained.bincode") {
        eprintln!("Failed to save model: {}", e);
    } else {
        println!("\nModel saved successfully!");
    }

    // Save training plot
    if let Err(e) = training_history.plot("training_progress.png") {
        eprintln!("Failed to save training plot: {}", e);
    } else {
        println!("Training plot saved to training_progress.png");
    }

    metrics_history
}
// Helper function for prediction
pub fn predict(model: &NeuralNetwork, x: &Array2<f64>) -> Array2<f64> {
    let hidden_input1 = x.dot(&model.weights1) + &model.bias1;
    let hidden_output1 = relu(&hidden_input1);

    let hidden_input2 = hidden_output1.dot(&model.weights2) + &model.bias2;
    let hidden_output2 = relu(&hidden_input2);

    let output_input = hidden_output2.dot(&model.weights3) + &model.bias3;
    relu(&output_input)
}

// Improved progress printing
fn print_progress(epoch: usize, total_epochs: usize, metrics: &TrainingMetrics) {
    let progress = (epoch as f64 / total_epochs as f64) * 100.0;
    print!("\rEpoch {:4}/{} ({:.1}%) | Train Loss: {:.4} | ", epoch, total_epochs, progress, metrics.train_loss);
    
    if let (Some(loss), Some(acc)) = (metrics.test_loss, metrics.test_accuracy) {
        print!(" | Test Loss: {:.4} | Test Acc: {:.2}%  | ",  loss, acc * 100.0);
    }
    
    if epoch == total_epochs - 1 {
        println!(); 
    }
}

pub fn cross_entropy_loss(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
    let epsilon = 1e-15;
    -(y_true * y_pred.mapv(|v| (v + epsilon).ln())).sum() / y_pred.shape()[0] as f64
}