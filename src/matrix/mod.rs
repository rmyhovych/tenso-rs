use std::{
    fmt::{Display, Write},
    result,
};

use rand::distributions::{Distribution, Normal};

use self::chunk::MatrixChunk;

mod chunk;

pub struct Matrix {
    size: (usize, usize),
    chunk_size: (usize, usize),
    chunks: Vec<MatrixChunk>,
}

impl Matrix {
    pub fn zero(size: (usize, usize)) -> Self {
        debug_assert_ne!(size.0 * size.1, 0);

        let chunk_size = MatrixChunk::get_chunk_size(size);
        Self {
            size,
            chunk_size,
            chunks: (0..(chunk_size.0 * chunk_size.1))
                .map(|_| MatrixChunk::new())
                .collect(),
        }
    }

    pub fn randn(size: (usize, usize), mean: f32, std_dev: f32) -> Self {
        let distribution = Normal::new(mean as f64, std_dev as f64);
        let rng = &mut rand::thread_rng();

        let chunk_size = MatrixChunk::get_chunk_size(size);
        Self {
            size,
            chunk_size,
            chunks: (0..(chunk_size.0 * chunk_size.1))
                .map(|i| {
                    let mut chunk = MatrixChunk::new();
                    let chunk_index = (i / chunk_size.1, i % chunk_size.1);
                    let element_limit = MatrixChunk::get_element_size()
                    let mut sample_function = |_| distribution.sample(rng) as f32;
                    chunk.for_each_mut(&mut sample_function);
                    chunk
                })
                .collect(),
        }
    }

    pub fn set(&mut self, index: (usize, usize), value: f32) {
        let chunk_index = MatrixChunk::get_chunk_index(index);
        let chunk = self.get_chunk_mut(chunk_index);
        chunk.set_unbounded(index, value);
    }

    pub fn get(&self, index: (usize, usize)) -> f32 {
        let chunk_index = MatrixChunk::get_chunk_index(index);
        let chunk = self.get_chunk(chunk_index);
        chunk.get_unbounded(index)
    }

    /* ------------------------------------------------- */

    pub fn unordered_reduce_operation<TFuncType: Fn(f32, f32) -> f32>(
        &self,
        func: TFuncType,
    ) -> Self {
        let mut result = Self::zero((1, 1));
        let mut total_value = 0.0;
        let mut reducer_fn = |v| total_value = func(total_value, v);

        for chunk in &self.chunks {
            chunk.foreach(&mut reducer_fn);
        }

        result.set((0, 0), total_value);
        result
    }

    pub fn unary_operation<TFuncType>(&self, func: TFuncType) -> Self
    where
        TFuncType: Fn(f32) -> f32,
    {
        let mut result = Self::zero(self.size);
        for i in 0..self.chunks.len() {
            result.chunks[i] = self.chunks[i].unary_operation(&func);
        }

        result
    }

    pub fn binary_operation<TFuncType>(&self, other: &Self, func: TFuncType) -> Self
    where
        TFuncType: Fn(f32, f32) -> f32,
    {
        debug_assert_eq!(self.size, other.size);

        let mut result = Self::zero(self.size);
        for i in 0..self.chunks.len() {
            result.chunks[i] = self.chunks[i].binary_operation(&other.chunks[i], &func);
        }

        result
    }

    pub fn add(&self, other: &Self) -> Self {
        self.binary_operation(other, |v0, v1| v0 + v1)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.binary_operation(other, |v0, v1| v0 - v1)
    }

    pub fn sum(&self) -> Self {
        self.unordered_reduce_operation(|total, v| total + v)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::zero((self.size.1, self.size.0));

        for y in 0..self.chunk_size.0 {
            for x in 0..self.chunk_size.1 {
                let chunk = self.get_chunk((y, x));
                result.chunks[x * self.chunk_size.0 + y] = chunk.transpose();
            }
        }

        result
    }

    pub fn matmul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.size.1, other.size.0);

        let mut result = Self::zero((self.size.0, other.size.1));
        for chunk_y in 0..self.chunk_size.0 {
            for chunk_x in 0..other.chunk_size.1 {
                let result_chunk = result.get_chunk_mut((chunk_y, chunk_x));
                for chunk_i in 0..self.chunk_size.1 {
                    let chunk_left = self.get_chunk((chunk_y, chunk_i));
                    let chunk_right = other.get_chunk((chunk_i, chunk_x));
                    let mm_result = chunk_left.matmul(&chunk_right);
                    result_chunk.binary_assign_operation(&mm_result, &|v0, v1| v0 + v1);
                }
            }
        }
        result
    }

    /* ------------------------------------------------- */

    fn for_each_value<TFuncType: FnMut(f32)>(&self, func: &mut TFuncType) {
        for chunk_y in 0..self.chunk_size.0 {
            for chunk_x in 0..self.chunk_size.1 {
                let max_val = if chunk_y == self.chunk_size.0 - 1 || chunk_x == self.chunk_size.1 - 1 {
                    let size = MatrixChunk::get_element_size((y, x));
                    (self.size.0 - size.0, self.size.1 - size.1)
                } else {
                    MatrixChunk::get_element_size((1, 1))
                };
            }
        }
    }

    fn get_chunk(&self, chunk_index: (usize, usize)) -> &MatrixChunk {
        &self.chunks[chunk_index.0 * self.chunk_size.1 + chunk_index.1]
    }

    fn get_chunk_mut(&mut self, chunk_index: (usize, usize)) -> &mut MatrixChunk {
        &mut self.chunks[chunk_index.0 * self.chunk_size.1 + chunk_index.1]
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const NUMBER_SPACING: u32 = 2;

        let number_dimension_fn = |num: f32| -> u32 {
            let mut number = num.floor() as i64;
            if number == 0 {
                1
            } else {
                let mut dimension = if number < 0 { 1 } else { 0 };
                number = number.abs();
                while number > 0 {
                    dimension += 1;
                    number /= 10;
                }

                dimension
            }
        };

        let mut max_dimension = 0;
        for y in 0..self.size.0 {
            for x in 0..self.size.1 {
                let number = self.get((y, x));
                let num_dimension = number_dimension_fn(number);
                max_dimension = max_dimension.max(num_dimension);
            }
        }

        let line_character_count =
            self.size.1 as u32 * (max_dimension + 3 + NUMBER_SPACING) + NUMBER_SPACING;

        let mut result =
            String::with_capacity(((line_character_count + 3) * (self.size.0 as u32 + 2)) as usize);

        result += " ";
        for _ in 0..line_character_count {
            result += "-";
        }
        result += " \n";

        for y in 0..self.size.0 {
            result += "|";
            for _ in 0..NUMBER_SPACING {
                result += " ";
            }

            for x in 0..self.size.1 {
                let number = self.get((y, x));
                let dimension = number_dimension_fn(number);
                write!(result, "{:.2}", number)?;

                for _ in 0..NUMBER_SPACING {
                    result += " ";
                }

                for _ in 0..(max_dimension - dimension) {
                    result += " ";
                }
            }
            result += "|\n";
        }
        result += " ";
        for _ in 0..line_character_count {
            result += "-";
        }
        result += " \n";

        write!(f, "{}", result)
    }
}
