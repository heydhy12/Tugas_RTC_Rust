use ndarray::Array2;

pub fn accuracy(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
    let correct = y_pred
        .rows()
        .into_iter()
        .zip(y_true.rows().into_iter())
        .filter(|(pred_row, true_row)| {
            let predicted_class = pred_row
                .iter()
                .enumerate()
                .max_by(|a, &b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let true_class = true_row
                .iter()
                .enumerate()
                .max_by(|a, &b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            predicted_class == true_class
        })
        .count();

    correct as f64 / y_pred.shape()[0] as f64
}