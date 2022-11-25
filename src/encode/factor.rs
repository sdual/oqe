pub fn shrinkage_factor(num_case: i32, param: f32) -> f32 {
    num_case as f32 / (num_case as f32 + param)
}
