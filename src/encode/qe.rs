use std::collections::HashMap;

use nalgebra::DVector;

use crate::encode::accum::PosteriorProbAccumulator;
use crate::encode::accum::PriorProbAccumulator;

use super::factor::shrinkage_factor;

pub struct OnlineTargetStatEncoder {
    posterior_accum_maps: Vec<HashMap<String, PosteriorProbAccumulator>>,
    prior_accum: PriorProbAccumulator,
    param: f32,
}

impl OnlineTargetStatEncoder {
    pub fn new(cat_feature_dim: usize, param: f32) -> Self {
        let post_accum_maps: Vec<HashMap<String, PosteriorProbAccumulator>> =
            (0usize..cat_feature_dim).map(|_| HashMap::new()).collect();

        OnlineTargetStatEncoder {
            posterior_accum_maps: post_accum_maps,
            prior_accum: PriorProbAccumulator::new(),
            param,
        }
    }

    pub fn accum_transform(&mut self, cat_features: &DVector<String>, target: i32) -> Vec<f32> {
        let mut encoded_vector = Vec::new();

        for (feature_index, feature_value) in cat_features.iter().enumerate() {
            let posterior_accum = self.posterior_accum_maps[feature_index].get_mut(feature_value);
            match posterior_accum {
                Some(accum) => {
                    let factor = shrinkage_factor(accum.total_count, self.param);
                    let encoded_value =
                        factor * accum.prob() + (1.0 - factor) * self.prior_accum.prob();
                    encoded_vector.push(encoded_value);
                    accum.increment(target);
                }
                _ => {
                    let mut post_accum = PosteriorProbAccumulator::new();
                    post_accum.increment(target);
                    self.posterior_accum_maps[feature_index]
                        .insert(feature_value.clone(), post_accum);
                    let added_post_accum = &self.posterior_accum_maps[feature_index][feature_value];

                    let factor = shrinkage_factor(added_post_accum.total_count, self.param);
                    let encoded_value =
                        factor * added_post_accum.prob() + (1.0 - factor) * self.prior_accum.prob();
                    encoded_vector.push(encoded_value);
                }
            }
        }
        self.prior_accum.increment(target);
        encoded_vector
    }
}

#[cfg(test)]
mod test {
    use df::dataframe::reader::StringDataFrame;
    use nalgebra::DVector;

    use super::OnlineTargetStatEncoder;

    // #[test]
    // fn test_online_target_stat_encoder_transform() {
    //     let test_cat_vec = DVector::from_vec(vec!["a".to_string(), "b".to_string()]);
    //     let test_cat_vec2 = DVector::from_vec(vec!["c".to_string(), "b".to_string()]);
    //     let test_cat_vec3 = DVector::from_vec(vec!["e".to_string(), "d".to_string()]);
    //     let mut encoder = OnlineTargetStatEncoder::new(test_cat_vec.len());
    //     let target = 1;
    //     for _ in 0..100 {
    //         encoder.accum_transform(&test_cat_vec, target);
    //         encoder.accum_transform(&test_cat_vec2, target);
    //         encoder.accum_transform(&test_cat_vec3, target);
    //     }
    // }

    #[test]
    fn test_with_titanic_dataset() {
        let feature_dim = 10;
        let feature_filepath = "/Users/qtk/work/python/titanic-data/titanic_categorical.csv";

        let has_headers = true;
        let all_df = StringDataFrame::read_csv(feature_filepath, has_headers, feature_dim);
        let mut encoder = OnlineTargetStatEncoder::new(feature_dim, 0.5);

        for (feature, label) in all_df.features.iter().zip(all_df.labels) {
            let result = encoder.accum_transform(&DVector::from_vec(feature.to_vec()), label);
            println!("{:?}", result);
        }
    }
}
