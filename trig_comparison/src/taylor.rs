use std::f64::consts::PI;

pub fn taylor_approximation(f: &str, x: f64) -> f64 {
    // Normalisasi sudut ke range -2π sampai 2π untuk akurasi yang lebih baik
    let x = x % (2.0 * PI);
    
    match f {
        "sin" => {
            // Deret Taylor untuk sin(x) hingga orde 7 (akurasi lebih baik)
            x - x.powi(3)/6.0 + x.powi(5)/120.0 - x.powi(7)/5040.0
        }
        "cos" => {
            // Deret Taylor untuk cos(x) hingga orde 8
            1.0 - x.powi(2)/2.0 + x.powi(4)/24.0 - x.powi(6)/720.0 + x.powi(8)/40320.0
        }
        _ => 0.0,
    }
}