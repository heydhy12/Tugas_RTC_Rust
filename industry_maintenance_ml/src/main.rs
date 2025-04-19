mod data;
mod models;
mod visualization;

use ndarray::{Array1, Array2, s};
use linfa::prelude::*;
use linfa::dataset::DatasetBase;
use rand::{thread_rng, seq::SliceRandom};
use data::loader::{load_data, normalize_data};
use models::svm::SVMModel;
use models::knn::{knn_predict_with_neighbors, knn_predict};
use visualization::plotter::{plot_svm_neighbors, plot_knn_neighbors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load and prepare data
    let dataset = load_data("csv/industry_maintenance.csv")?;
    let normalized_features = normalize_data(dataset.features);
    let dataset = DatasetBase::new(normalized_features, dataset.labels);
    let (train, test) = dataset.split_with_ratio(0.8);

    // Train and Accuracy SVM
    let svm_model = SVMModel::train(&train)?;
    let svm_accuracy = svm_model.evaluate_accuracy(&test);
    println!("Akurasi Model SVM: {:.2}%", svm_accuracy * 100.0);

    // Train KNN
    let (knn_pred, neighbors) = knn_predict_with_neighbors(
        train.records(),
        train.targets(),
        test.records(),
        5  // Jumlah
    );

    // Accuracy KNN
    let knn_accuracy = evaluate_accuracy(&knn_pred, test.targets());
    println!("Akurasi Model KNN: {:.2}%", knn_accuracy * 100.0);
    
    fn evaluate_accuracy(predictions: &Array1<usize>, actual: &Array1<usize>) -> f64 {
    let correct = predictions
        .iter()
        .zip(actual.iter())
        .filter(|(pred, actual)| pred == actual)
        .count();
    correct as f64 / actual.len() as f64
    }
    
    // Plot SVM results
    let test_sample = test.records().slice(s![0..1, ..]).to_owned();
    let svm_pred = svm_model.predict(&test_sample);
    plot_svm_neighbors(
        train.records(),
        train.targets(),
        &test_sample,
        svm_pred[0],
        "svm_neighbors.png",
    )?;

    // Plot KNN results
    plot_knn_neighbors(
    train.records(),
    train.targets(),
    &test.records().slice(s![0..1, ..]).to_owned(),  // ambil satu sampel
    knn_pred[0],
    &neighbors[0],  
    "knn_neighbors.png"
    )?;

    // Show predictions for 10 random samples
    show_sample_predictions(&train, &test, &svm_model)?;

    Ok(())
}

fn show_sample_predictions(
    train: &DatasetBase<Array2<f64>, Array1<usize>>,
    test: &DatasetBase<Array2<f64>, Array1<usize>>,
    svm_model: &SVMModel,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..test.nsamples()).collect();
    indices.shuffle(&mut rng);
    let selected_indices = &indices[0..10.min(indices.len())];

    let test_samples = test.records().select(ndarray::Axis(0), selected_indices);
    let test_labels = test.targets().select(ndarray::Axis(0), selected_indices);

    let svm_predictions = svm_model.predict(&test_samples);
    let knn_predictions = knn_predict(train.records(), train.targets(), &test_samples, 5);

    println!("\nHasil Prediksi untuk 10 Sampel dari Data Test:");
    for (i, ((svm_pred, knn_pred), actual)) in svm_predictions.iter()
        .zip(knn_predictions.iter())
        .zip(test_labels.iter())
        .enumerate()
    {
        let svm_category = get_failure_type(*svm_pred);
        let knn_category = get_failure_type(*knn_pred);
        let actual_category = get_failure_type(*actual);

        println!(
            "Sampel {}: Prediksi SVM = {}, Prediksi KNN = {}, Aktual = {}",
            i + 1, svm_category, knn_category, actual_category
        );
    }

    Ok(())
}

fn get_failure_type(label: usize) -> &'static str {
    match label {
        0 => "No Failure",
        1 => "Heat Dissipation Failure",
        2 => "Overstrain Failure",
        3 => "Power Failure",
        _ => "Unknown",
    }
}