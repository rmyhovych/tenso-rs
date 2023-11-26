use std::ops::*;

use super::Matrix;

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        self.binary_operation(rhs, |v0, v1| v0 + v1)
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        self.binary_operation(rhs, |v0, v1| v0 - v1)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.binary_operation(rhs, |v0, v1| v0 * v1)
    }
}

impl Mul<&Matrix> for f32 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        rhs.unary_operation(|v| self * v)
    }
}

impl Mul<f32> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Self::Output {
        self.unary_operation(|v| rhs * v)
    }
}

impl Matrix {
    pub fn sum(&self) -> Self {
        self.unordered_reduce_operation(|total, v| total + v)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new_zero([self.size[1], self.size[0]]);

        for y in 0..self.chunk_size[0] {
            for x in 0..self.chunk_size[1] {
                let chunk = self.get_chunk([y, x]);
                result.chunks[x * self.chunk_size[0] + y] = chunk.transpose();
            }
        }

        result
    }

    pub fn matmul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.size[1], other.size[0]);

        let mut result = Self::new_zero([self.size[0], other.size[1]]);
        for chunk_y in 0..self.chunk_size[0] {
            for chunk_x in 0..other.chunk_size[1] {
                let result_chunk = result.get_chunk_mut([chunk_y, chunk_x]);
                for chunk_i in 0..self.chunk_size[1] {
                    let chunk_left = self.get_chunk([chunk_y, chunk_i]);
                    let chunk_right = other.get_chunk([chunk_i, chunk_x]);
                    let mm_result = chunk_left.matmul(&chunk_right);
                    result_chunk.binary_assign_operation(&mm_result, &|v0, v1| v0 + v1);
                }
            }
        }
        result
    }
}
