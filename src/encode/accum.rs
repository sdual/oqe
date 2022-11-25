pub struct PosteriorProbAccumulator {
    target_value_count: i64,
    target_value: String,
}

impl PosteriorProbAccumulator {
    pub fn new(target_value: String) -> Self {
        PosteriorProbAccumulator {
            target_value_count: 0,
            target_value,
        }
    }

    pub fn prob(&self, total_count: i64) -> f32 {
        if total_count == 0 {
            0.0
        } else {
            (self.target_value_count as f64 / total_count as f64) as f32
        }
    }

    pub fn increment(&mut self) {
        self.target_value_count += 1;
    }
}

pub struct PriorProbAccumulator {
    total_count: i64,
    positive_target_count: i64,
}

impl PriorProbAccumulator {
    pub fn new() -> Self {
        PriorProbAccumulator {
            total_count: 0,
            positive_target_count: 0,
        }
    }

    pub fn prob(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.positive_target_count as f32 / self.total_count as f32
        }
    }

    pub fn increment(&mut self, target: i32) {
        self.total_count += 1;
        if target == 1 {
            self.positive_target_count += 1;
        }
    }

    pub fn get_total_count(&self) -> i64 {
        self.total_count
    }
}
