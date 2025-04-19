use csv::Reader;
use ndarray::Array2;
use serde::Deserialize;
use std::fs::File;
use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;


#[derive(Debug, Deserialize)]
pub struct Record {
    pub air_temperature_k: f64,
    pub process_temperature_k: f64,
    pub rotational_speed_rpm: f64,
    pub torque_nm: f64,
    pub failure_type: String,
}

pub struct Dataset {
    pub features: Array2<f64>,
    pub labels: Array2<f64>,
}

pub struct SplitDataset {
    pub train_features: Array2<f64>,
    pub train_labels: Array2<f64>,
    pub test_features: Array2<f64>,
    pub test_labels: Array2<f64>,
}

impl Dataset {
    pub fn from_csv(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(file_path)?;
        let mut rdr = Reader::from_reader(file);
        
        let mut records: Vec<Record> = Vec::new();
        for result in rdr.deserialize() {
            records.push(result?);
        }
        
        let mut features = Array2::zeros((records.len(), 4));
        let mut labels = Array2::zeros((records.len(), 4));
        
        for (i, record) in records.iter().enumerate() {
            features[[i, 0]] = record.air_temperature_k;
            features[[i, 1]] = record.process_temperature_k;
            features[[i, 2]] = record.rotational_speed_rpm;
            features[[i, 3]] = record.torque_nm;
            
            match record.failure_type.as_str() {
                "Power Failure" => labels[[i, 0]] = 1.0,
                "Overstrain Failure" => labels[[i, 1]] = 1.0,
                "No Failure" => labels[[i, 2]] = 1.0,
                "Heat Dissipation Failure" => labels[[i, 3]] = 1.0,
                _ => (),
            }
        }
        
        Ok(Dataset { features, labels })
    }

    pub fn split_data(&self, test_size: f64, shuffle: bool) -> SplitDataset {
        assert!(
            test_size > 0.0 && test_size < 1.0,
        );
        
        let n_samples = self.features.nrows();
        let n_test = (n_samples as f64 * test_size) as usize;
        let n_train = n_samples - n_test;

        // Buat indeks dan acak data
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }

        // Split features
        let train_features = self.features.select(ndarray::Axis(0), &indices[..n_train]);
        let test_features = self.features.select(ndarray::Axis(0), &indices[n_train..]);

        // Split labels
        let train_labels = self.labels.select(ndarray::Axis(0), &indices[..n_train]);
        let test_labels = self.labels.select(ndarray::Axis(0), &indices[n_train..]);

        SplitDataset {
            train_features,
            train_labels,
            test_features,
            test_labels,
        }
    }

}