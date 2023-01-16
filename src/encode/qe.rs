use nalgebra::{DVector, SVector};
use fnv::FnvHashMap;

use crate::encode::accum::PosteriorProbAccumulator;
use crate::encode::accum::PriorProbAccumulator;

use super::factor::list_shrinkage_factor;
use super::factor::shrinkage_factor;

pub struct OnlineTargetStatEncoder {
    posterior_accum_maps: Vec<FnvHashMap<String, PosteriorProbAccumulator>>,
    prior_accum: PriorProbAccumulator,
    param: f32,
}

impl OnlineTargetStatEncoder {
    pub fn new(cat_feature_dim: usize, param: f32) -> Self {
        let post_accum_maps: Vec<FnvHashMap<String, PosteriorProbAccumulator>> =
            (0usize..cat_feature_dim).map(|_| FnvHashMap::default()).collect();

        OnlineTargetStatEncoder {
            posterior_accum_maps: post_accum_maps,
            prior_accum: PriorProbAccumulator::new(),
            param,
        }
    }

    pub fn accum_transform(&mut self, cat_features: &Vec<String>, target: i32) -> Vec<f32> {
        let mut encoded_vector = Vec::new();

        for (feature_index, feature_value) in cat_features.iter().enumerate() {
            let posterior_accum = self.posterior_accum_maps[feature_index].get_mut(feature_value);
            match posterior_accum {
                Some(accum) => {
                    let factor = shrinkage_factor(accum.total_count, self.param);
                    let encoded_value =
                        factor * accum.prob(); // + (1.0 - factor) * self.prior_accum.prob();
                    encoded_vector.push(encoded_value);
                    accum.increment(target);
                }
                _ => {
                    let post_accum = PosteriorProbAccumulator::new();
                    self.posterior_accum_maps[feature_index]
                        .insert(feature_value.clone(), post_accum);
                    let added_post_accum = self.posterior_accum_maps[feature_index]
                        .get_mut(feature_value)
                        .unwrap();

                    let factor = shrinkage_factor(added_post_accum.total_count, self.param);
                    let encoded_value =
                        factor * added_post_accum.prob(); // (1.0 - factor) * self.prior_accum.prob();
                    encoded_vector.push(encoded_value);
                    added_post_accum.increment(target);
                }
            }
        }
        self.prior_accum.increment(target);
        encoded_vector
    }
}

pub struct OnlineListTargetStatEncoder {
    posterior_accum_maps: Vec<FnvHashMap<String, PosteriorProbAccumulator>>,
    prior_accum: PriorProbAccumulator,
    param: f32,
}

impl OnlineListTargetStatEncoder {
    pub fn new(cat_list_feature_dim: usize, param: f32) -> Self {
        let post_accum_maps: Vec<FnvHashMap<String, PosteriorProbAccumulator>> = (0usize
            ..cat_list_feature_dim)
            .map(|_| FnvHashMap::default())
            .collect();

        OnlineListTargetStatEncoder {
            posterior_accum_maps: post_accum_maps,
            prior_accum: PriorProbAccumulator::new(),
            param,
        }
    }

    pub fn accum_transform(
        &mut self,
        cat_list_features: &Vec<Vec<String>>,
        target: i32,
    ) -> Vec<f32> {
        let mut encoded_vector = Vec::new();

        for (feature_index, feature_list) in cat_list_features.iter().enumerate() {
            let mut encoded_value = 0.0;
            let mut total_factor = 0.0;

            for feature_value in feature_list {
                let posterior_accum =
                    self.posterior_accum_maps[feature_index].get_mut(feature_value);
                match posterior_accum {
                    Some(accum) => {
                        // let factor = shrinkage_factor(
                        //     accum.total_count,
                        //     self.param,
                        // );
                        let factor = 1.0;
                        encoded_value += factor * accum.prob();
                        total_factor += factor;
                        accum.increment(target);
                    }
                    _ => {
                        let post_accum = PosteriorProbAccumulator::new();
                        self.posterior_accum_maps[feature_index]
                            .insert(feature_value.clone(), post_accum);
                        let added_post_accum = self.posterior_accum_maps[feature_index]
                            .get_mut(feature_value)
                            .unwrap();

                        // let factor = shrinkage_factor(
                        //     added_post_accum.total_count,
                        //     self.param,
                        // );
                        let factor = 1.0;
                        encoded_value += factor * added_post_accum.prob();
                        total_factor += factor;
                        added_post_accum.increment(target);
                    }
                }
            }
            // encoded_value += (1.0 - total_factor) * self.prior_accum.prob();
            encoded_vector.push(encoded_value);
        }
        self.prior_accum.increment(target);
        encoded_vector
    }
}

// pub struct OnlineMapTargetStatEncoder {
//     posterior_accum_maps: Vec<HashMap<String, PosteriorProbAccumulator>>,
//     prior_accum: PriorProbAccumulator,
//     param: f32,
// }

// impl OnlineMapTargetStatEncoder {
//     pub fn new(cat_map_feature_dim: usize, param: f32) -> Self {
//         let post_accum_maps: Vec<HashMap<String, PosteriorProbAccumulator>> = (0usize
//             ..cat_map_feature_dim)
//             .map(|_| HashMap::new())
//             .collect();

//         OnlineMapTargetStatEncoder {
//             posterior_accum_maps: post_accum_maps,
//             prior_accum: PriorProbAccumulator::new(),
//             param,
//         }
//     }

//     pub fn accum_transform(&mut self, cat_fatures: &DVector<String>, target: i32) -> Vec<f32> {
//         let encoded_vector = Vec::new();

//     }
// }

#[cfg(test)]
mod test {
    use df::dataframe::reader::StringDataFrame;

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
        let feature_filepath = "./titanic_categorical.csv";

        let has_headers = true;
        let all_df = StringDataFrame::read_csv(feature_filepath, has_headers, feature_dim);
        let mut encoder = OnlineTargetStatEncoder::new(feature_dim, 10.0);

        for (feature, label) in all_df.features.iter().zip(all_df.labels) {
            let result = encoder.accum_transform(&feature, label);
            println!("{:?}", result);
        }
    }
}
