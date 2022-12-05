pub fn shrinkage_factor(num_case: i64, param: f32) -> f32 {
    (num_case as f64 / (num_case as f64 + param as f64)) as f32
}

pub fn list_shrinkage_factor(num_case: i64, total_count: i64, param: f32) -> f32 {
    (num_case as f64 / (total_count as f64 + param as f64)) as f32
}
