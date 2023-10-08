use ordered_float::NotNan;
use std::slice::Iter;

pub struct MinStore {
    values: [NotNan<f32>; 5],
    n: usize,
}

impl MinStore {
    pub(crate) fn new() -> Self {
        MinStore {
            values: [NotNan::default(); 5],
            n: 0,
        }
    }

    pub(crate) fn push(&mut self, value: NotNan<f32>) {
        if self.n < 5 {
            self.values[self.n] = value;
            self.n += 1;
        } else {
            let max_idx = self
                .values
                .iter()
                .enumerate()
                .max_by_key(|(_, &value)| value)
                .map(|(idx, _)| idx)
                .unwrap();
            if value < self.values[max_idx] {
                self.values[max_idx] = value;
            }
        }
    }

    fn iter(&self) -> Iter<'_, NotNan<f32>> {
        self.values[..self.n].iter()
    }

    pub(crate) fn min(&self) -> f32 {
        if self.n == 0 {
            return f32::NAN;
        }
        self.iter().min().unwrap().into_inner()
    }

    pub(crate) fn mean(&self) -> f32 {
        if self.n == 0 {
            return f32::NAN;
        }
        self.iter().sum::<NotNan<f32>>().into_inner() / self.n as f32
    }
}
