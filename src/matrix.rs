use std::fmt::Display;

use rand::{distributions::Normal, Rng};

pub struct Matrix {
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(height: usize, width: usize) -> Self {
        Self::from_const(height, width, 0.0)
    }

    pub fn from_const(height: usize, width: usize, value: f32) -> Self {
        Self {
            height,
            width,
            data: (0..height * width).map(|_| value).collect(),
        }
    }

    pub fn randn(height: usize, width: usize, mean: f64, std: f64) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std);

        Self {
            height,
            width,
            data: (0..height * width)
                .map(|_| rng.sample(&normal) as f32)
                .collect(),
        }
    }

    pub fn get(&self, y: usize, x: usize) -> f32 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, val: f32, y: usize, x: usize) {
        self.data[y * self.width + x] = val;
    }
}

impl Display for Matrix {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                if let Err(e) = fmt.write_str(format!("| {:.1$}\t", self.get(y, x), 2).as_str()) {
                    return Err(e);
                }
            }
            if let Err(e) = fmt.write_str("|\n") {
                return Err(e);
            }
        }
        Ok(())
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            data: self.data.clone(),
        }
    }
}
