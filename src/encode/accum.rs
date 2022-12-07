pub struct PosteriorProbAccumulator {
    pub total_count: i64,
    target_value_count: i64,
}

impl PosteriorProbAccumulator {
    pub fn new() -> Self {
        PosteriorProbAccumulator {
            total_count: 0,
            target_value_count: 0,
        }
    }

    pub fn prob(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.target_value_count as f64 / self.total_count as f64) as f32
        }
    }

    pub fn increment(&mut self, target: i32) {
        self.total_count += 1;
        if target == 1 {
            self.target_value_count += 1;
        }
    }
}

pub struct PriorProbAccumulator {
    pub total_count: i64,
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
