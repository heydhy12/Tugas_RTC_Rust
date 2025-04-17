use std::io;
mod table;
mod taylor;
use table::TrigTable;
use taylor::taylor_approximation;

fn main() {
    // 1. Buat lookup table (sekarang mencakup semua sudut 0-360°)
    let trig_table = TrigTable::new();
    
    // 2. Input pengguna
    let function = loop {
        let input = get_user_input("Masukkan fungsi (sin/cos):").to_lowercase();
        if input == "sin" || input == "cos" {
            break input;
        }
        println!("Input tidak valid. Harap masukkan 'sin' atau 'cos'.");
    };
    
    let angle_type = loop {
        let input = get_user_input("Masukkan jenis sudut (radian/derajat):").to_lowercase();
        if input == "radian" || input == "derajat" {
            break input;
        }
        println!("Input tidak valid. Harap masukkan 'radian' atau 'derajat'.");
    };
    
    let x = loop {
        let input = get_user_input("Masukkan nilai x:");
        match input.parse::<f64>() {
            Ok(num) => break num,
            Err(_) => println!("Harap masukkan angka yang valid."),
        }
    };
    
    // 3. Konversi ke radian jika input derajat
    let (x_rad, degrees) = if angle_type == "derajat" {
        let deg = x.round() as u32;
        (degrees_to_radians(x), deg)
    } else {
        let deg = (x * 180.0 / std::f64::consts::PI).round() as u32;
        (x, deg)
    };
    
    // Normalisasi derajat ke 0-360
    let degrees_normalized = degrees % 360;
    
    println!("\nSudut: {}° ({:.4} rad)", degrees_normalized, x_rad);
    
    // 4. Hitung dengan Taylor (orde lebih tinggi)
    let taylor_val = taylor_approximation(&function, x_rad);
    
    // 5. Ambil nilai dari lookup table (selalu tersedia sekarang)
    let table_val = if function == "sin" {
        trig_table.get_sin(degrees_normalized)
    } else {
        trig_table.get_cos(degrees_normalized)
    };
    
    // 6. Tampilkan hasil
    println!("\n--- Hasil ---");
    println!("Deret Taylor: {}({}°) = {:.6}", function, degrees_normalized, taylor_val);
    
    if let Some(t_val) = table_val {
        println!("Lookup Table: {}({}°) = {:.6}", function, degrees_normalized, t_val);
        println!("\n--- Selisih ---");
        println!("Error: {:.6e}", (taylor_val - t_val).abs());
    }
}

fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

fn get_user_input(prompt: &str) -> String {
    println!("{}", prompt);
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Gagal membaca input");
    input.trim().to_string()
}