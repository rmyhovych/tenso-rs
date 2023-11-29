use std::{
    fmt::{Display, Write},
    ops::{Index, IndexMut},
};

use rand::distributions::{Distribution, Normal};

use self::chunk::MatrixChunk;

mod chunk;
pub mod op;

pub struct Matrix {
    size: [usize; 2],
    chunk_size: [usize; 2],
    chunks: Vec<MatrixChunk>,
}

impl Matrix {
    pub fn new_zero(size: [usize; 2]) -> Self {
        assert_ne!(size[0] * size[1], 0);

        let chunk_size = MatrixChunk::get_chunk_size(size);
        Self {
            size,
            chunk_size,
            chunks: (0..(chunk_size[0] * chunk_size[1]))
                .map(|_| MatrixChunk::new())
                .collect(),
        }
    }

    pub fn new_randn(size: [usize; 2], mean: f32, std_dev: f32) -> Self {
        let distribution = Normal::new(mean as f64, std_dev as f64);
        let rng = &mut rand::thread_rng();

        let mut result = Self::new_zero(size);
        result.for_each_value_mut(&mut |_| distribution.sample(rng) as f32);

        result
    }

    pub fn new_value(size: [usize; 2], value: f32) -> Self {
        let mut result = Self::new_zero(size);
        result.for_each_value_mut(&mut |_| value);
        result
    }

    pub fn size(&self) -> [usize; 2] {
        self.size
    }

    pub fn take_clear(&mut self) -> Self {
        let chunk_copy = self.chunks.clone();
        self.chunks.iter_mut().for_each(|c| c.clear());

        Self {
            size: self.size,
            chunk_size: self.chunk_size,
            chunks: chunk_copy,
        }
    }

    /* ------------------------------------------------- */

    pub fn unordered_reduce_operation<TFuncType: Fn(f32, f32) -> f32>(
        &self,
        func: TFuncType,
    ) -> Self {
        let mut total_value = 0.0;
        self.for_each_value(&mut |v| total_value = func(total_value, v));

        let mut result = Self::new_zero([1, 1]);
        result[[0, 0]] = total_value;
        result
    }

    pub fn unary_operation<TFuncType>(&self, func: TFuncType) -> Self
    where
        TFuncType: Fn(f32) -> f32,
    {
        let mut result = Self::new_zero(self.size);
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

        let mut result = Self::new_zero(self.size);
        for i in 0..self.chunks.len() {
            result.chunks[i] = self.chunks[i].binary_operation(&other.chunks[i], &func);
        }

        result
    }

    /* ------------------------------------------------- */

    fn for_each_value<TFuncType: FnMut(f32)>(&self, func: &mut TFuncType) {
        for chunk_y in 0..self.chunk_size[0] {
            for chunk_x in 0..self.chunk_size[1] {
                let max_val = self.get_chunk_element_size([chunk_y, chunk_x]);

                let chunk = self.get_chunk([chunk_y, chunk_x]);
                chunk.for_each(&mut |coord, val| {
                    if coord[0] < max_val[0] && coord[1] < max_val[1] {
                        func(val);
                    }
                });
            }
        }
    }

    fn for_each_value_mut<TFuncType: FnMut(f32) -> f32>(&mut self, func: &mut TFuncType) {
        for chunk_y in 0..self.chunk_size[0] {
            for chunk_x in 0..self.chunk_size[1] {
                let max_val = self.get_chunk_element_size([chunk_y, chunk_x]);

                let chunk = self.get_chunk_mut([chunk_y, chunk_x]);
                chunk.for_each_mut(&mut |coord, val| {
                    if coord[0] < max_val[0] && coord[1] < max_val[1] {
                        func(val)
                    } else {
                        0.0
                    }
                });
            }
        }
    }

    fn get_chunk(&self, chunk_index: [usize; 2]) -> &MatrixChunk {
        &self.chunks[chunk_index[0] * self.chunk_size[1] + chunk_index[1]]
    }

    fn get_chunk_mut(&mut self, chunk_index: [usize; 2]) -> &mut MatrixChunk {
        &mut self.chunks[chunk_index[0] * self.chunk_size[1] + chunk_index[1]]
    }

    fn get_chunk_element_size(&self, chunk_index: [usize; 2]) -> [usize; 2] {
        if chunk_index[0] == self.chunk_size[0] - 1 || chunk_index[1] == self.chunk_size[1] - 1 {
            let size = MatrixChunk::get_element_size(chunk_index);
            [self.size[0] - size[0], self.size[1] - size[1]]
        } else {
            MatrixChunk::get_element_size([1, 1])
        }
    }
}

impl Index<[usize; 2]> for Matrix {
    type Output = f32;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let chunk_index = MatrixChunk::get_chunk_index(index);
        let chunk = self.get_chunk(chunk_index);
        chunk.get_unbounded(index)
    }
}

impl IndexMut<[usize; 2]> for Matrix {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let chunk_index = MatrixChunk::get_chunk_index(index);
        let chunk = self.get_chunk_mut(chunk_index);
        chunk.get_unbounded_mut(index)
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            chunk_size: self.chunk_size,
            chunks: self.chunks.clone(),
        }
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
        for y in 0..self.size[0] {
            for x in 0..self.size[1] {
                let number = self[[y, x]];
                let num_dimension = number_dimension_fn(number);
                max_dimension = max_dimension.max(num_dimension);
            }
        }

        let line_character_count =
            self.size[1] as u32 * (max_dimension + 3 + NUMBER_SPACING) + NUMBER_SPACING;

        let mut result = String::with_capacity(
            ((line_character_count + 3) * (self.size[0] as u32 + 2)) as usize,
        );

        result += " ";
        for _ in 0..line_character_count {
            result += "-";
        }
        result += " \n";

        for y in 0..self.size[0] {
            result += "|";
            for _ in 0..NUMBER_SPACING {
                result += " ";
            }

            for x in 0..self.size[1] {
                let number = self[[y, x]];
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
