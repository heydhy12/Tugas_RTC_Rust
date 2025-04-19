use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;  // Import trait RandomExt
use ndarray_rand::rand_distr::Uniform;
use crate::model::activations::{relu, softmax};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};
use bincode;

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array1<f64>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size1: usize, hidden_size2: usize, output_size: usize) -> Self {
        let he_std1 = (2.0 / input_size as f64).sqrt();
        let he_std2 = (2.0 / hidden_size1 as f64).sqrt();
        let he_std3 = (2.0 / hidden_size2 as f64).sqrt();

        let weights1 = Array2::random(
            (input_size, hidden_size1),
            Uniform::new(-he_std1, he_std1)
        );
        
        let bias1 = Array1::zeros(hidden_size1);

        let weights2 = Array2::random(
            (hidden_size1, hidden_size2),
            Uniform::new(-he_std2, he_std2)
        );
        
        let bias2 = Array1::zeros(hidden_size2);

        let weights3 = Array2::random(
            (hidden_size2, output_size),
            Uniform::new(-he_std3, he_std3)
        );
        
        let bias3 = Array1::zeros(output_size);

        NeuralNetwork {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let hidden_input1 = x.dot(&self.weights1) + &self.bias1;
        let hidden_output1 = relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
        let hidden_output2 = relu(&hidden_input2);

        let output_input = hidden_output2.dot(&self.weights3) + &self.bias3;
        softmax(&output_input)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let encoded: Vec<u8> = bincode::serialize(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let decoded: NeuralNetwork = bincode::deserialize(&buffer).unwrap();
        Ok(decoded)
    }
}