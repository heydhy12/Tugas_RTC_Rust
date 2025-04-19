use ndarray::{Array2, Axis};

pub fn normalize_zscore(data: &mut Array2<f64>) {
    let mean = data.mean_axis(Axis(0)).unwrap();
    let std = data.std_axis(Axis(0), 0.0);
    
    // Broadcast operation untuk memastikan dimensi sesuai
    for mut row in data.rows_mut() {
        row -= &mean;
        row /= &std;
    }
}