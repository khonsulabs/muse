use crate::sampler::{FrameInfo, PreparedSampler, Sample, Sampler};

#[derive(Debug)]
pub struct Max {
    sources: Vec<PreparedSampler>,
}

impl Max {
    pub fn new(sources: Vec<PreparedSampler>) -> Self {
        Self { sources }
    }
}

impl Sampler for Max {
    fn sample(&mut self, frame: &FrameInfo) -> Option<Sample> {
        let mut result: Option<Sample> = None;
        for sample in self.sources.iter_mut().filter_map(|s| s.sample(frame)) {
            result = result
                .map(|mut existing| {
                    existing.left = if sample.left > existing.left {
                        sample.left
                    } else {
                        existing.left
                    };
                    existing.right = if sample.right > existing.right {
                        sample.right
                    } else {
                        existing.right
                    };
                    existing
                })
                .or(Some(sample))
        }
        result
    }
}
