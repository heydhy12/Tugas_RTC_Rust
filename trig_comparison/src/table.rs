use std::collections::HashMap;
use std::f64::consts::PI;

pub struct TrigTable {
    sin_data: HashMap<u32, f64>,  // Key: derajat, Value: sin
    cos_data: HashMap<u32, f64>,  // Key: derajat, Value: cos
}

impl TrigTable {
    pub fn new() -> Self {
        let mut sin_data = HashMap::new();
        let mut cos_data = HashMap::new();
        
        // Isi tabel dengan nilai sudut dari 0° sampai 360° dengan interval 1°
        for deg in 0..=360 {
            let rad = degrees_to_radians(deg as f64);
            sin_data.insert(deg, rad.sin());
            cos_data.insert(deg, rad.cos());
        }
        
        TrigTable { sin_data, cos_data }
    }
    
    pub fn get_sin(&self, degrees: u32) -> Option<f64> {
        // Gunakan periodisitas sinus (360°)
        let normalized_deg = degrees % 360;
        self.sin_data.get(&normalized_deg).copied()
    }
    
    pub fn get_cos(&self, degrees: u32) -> Option<f64> {
        // Gunakan periodisitas cosinus (360°)
        let normalized_deg = degrees % 360;
        self.cos_data.get(&normalized_deg).copied()
    }
}

fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}