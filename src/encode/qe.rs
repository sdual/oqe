use std::collections::HashMap;

use nalgebra::DVector;

use crate::encode::accum::PosteriorProbAccumulator;
use crate::encode::accum::PriorProbAccumulator;

pub struct OnlineTargetStatEncoder {
    posterior_accum_maps: Vec<HashMap<String, PosteriorProbAccumulator>>,
    prior_accum: PriorProbAccumulator,
}

impl OnlineTargetStatEncoder {
    pub fn new(cat_feature_dim: usize) -> Self {
        let post_accum_maps: Vec<HashMap<String, PosteriorProbAccumulator>> =
            (0usize..cat_feature_dim).map(|_| HashMap::new()).collect();

        OnlineTargetStatEncoder {
            posterior_accum_maps: post_accum_maps,
            prior_accum: PriorProbAccumulator::new(),
        }
    }

    pub fn accum_transform(&mut self, cat_features: &DVector<String>, target: i32) -> DVector<f32> {
        let encoded_vector = self.accumulate(cat_features, target);
        println!("{:?}", encoded_vector);
        DVector::from_vec(vec![])
    }

    fn accumulate(&mut self, cat_features: &DVector<String>, target: i32) -> Vec<f32> {
        let mut encoded_vector = Vec::new();

        for (feature_index, feature_value) in cat_features.iter().enumerate() {
            let posterior_accum = self.posterior_accum_maps[feature_index].get_mut(feature_value);
            match posterior_accum {
                Some(accum) => {
                    let encoded_value = accum.prob(self.prior_accum.get_total_count());
                    encoded_vector.push(encoded_value);
                    accum.increment();
                }
                _ => {
                    let mut post_accum = PosteriorProbAccumulator::new(feature_value.clone());

                    let encoded_value = post_accum.prob(self.prior_accum.get_total_count());
                    encoded_vector.push(encoded_value);

                    post_accum.increment();
                    self.posterior_accum_maps[feature_index]
                        .insert(feature_value.clone(), post_accum);
                }
            }
        }
        self.prior_accum.increment(target);
        encoded_vector
    }
}

#[cfg(test)]
mod test {
    use nalgebra::DVector;

    use super::OnlineTargetStatEncoder;

    #[test]
    fn test_online_quantile_encode_transform() {
        let test_cat_vec = DVector::from_vec(vec!["a".to_string(), "b".to_string()]);
        let test_cat_vec2 = DVector::from_vec(vec!["c".to_string(), "b".to_string()]);
        let test_cat_vec3 = DVector::from_vec(vec!["e".to_string(), "d".to_string()]);
        let mut encoder = OnlineTargetStatEncoder::new(test_cat_vec.len());
        let target = 1;
        for _ in 0..100 {
            encoder.accum_transform(&test_cat_vec, target);
            encoder.accum_transform(&test_cat_vec2, target);
            encoder.accum_transform(&test_cat_vec3, target);
        }
    }
}
