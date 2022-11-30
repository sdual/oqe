pub fn shrinkage_factor(num_case: i64, param: f32) -> f32 {
    (num_case as f64 / (num_case as f64 + param as f64)) as f32
}
