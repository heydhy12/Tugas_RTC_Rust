use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// Example usage of evaluate_accuracy in the main function
#[allow(dead_code)]
fn main() {
    use ndarray::array;

    let predictions = array![1, 0, 1, 1];
    let actual = array![1, 0, 0, 1];
    let accuracy = evaluate_accuracy(&predictions, &actual);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}

// Original knn_predict function
pub fn knn_predict(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    k: usize,
) -> Array1<usize> {
    let (predictions, _) = knn_predict_with_neighbors(train, train_labels, test, k);
    predictions
}

// New function that also returns neighbors
pub fn knn_predict_with_neighbors(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    k: usize,
) -> (Array1<usize>, Vec<Vec<usize>>) {
    let mut predictions = Array1::zeros(test.nrows());
    let mut neighbors_list = Vec::with_capacity(test.nrows());

    for (i, test_sample) in test.axis_iter(Axis(0)).enumerate() {
        let mut distances: Vec<(usize, f64)> = train.axis_iter(Axis(0))
            .enumerate()
            .map(|(idx, train_sample)| (idx, euclidean_distance(&train_sample.to_owned(), &test_sample.to_owned())))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nearest_indices: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        neighbors_list.push(nearest_indices.clone());

        let mut label_counts = HashMap::new();
        for &idx in &nearest_indices {
            let label = train_labels[idx];
            *label_counts.entry(label).or_insert(0) += 1;
        }

        predictions[i] = label_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or(0);
    }

    (predictions, neighbors_list)
}

pub fn evaluate_accuracy(
    predictions: &Array1<usize>,
    actual: &Array1<usize>
) -> f32 {
    predictions.iter()
        .zip(actual.iter())
        .filter(|(pred, true_val)| pred == true_val)
        .count() as f32 / predictions.len() as f32
}

// Example usage of evaluate_accuracy
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_evaluate_accuracy() {
        let predictions = array![1, 0, 1, 1];
        let actual = array![1, 0, 0, 1];
        let accuracy = evaluate_accuracy(&predictions, &actual);
        assert!((accuracy - 0.75).abs() < 1e-6);
    }
}