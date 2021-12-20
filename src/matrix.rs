use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Display,
    iter::Zip,
    ops::{Index, IndexMut},
    rc::Rc,
    slice::Iter,
};

use rand::{distributions::Normal, Rng};

#[derive(Clone)]
pub struct Matrix {
    height: usize,
    width: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn new(height: usize, width: usize, data: Vec<f32>) -> Self {
        Self {
            height,
            width,
            data,
        }
    }

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

    /*------------------------------------------------------*/

    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn set(&mut self, other: Matrix) {
        self.width = other.width;
        self.height = other.height;
        self.data = other.data;
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn chain_data<T>(&self, accessor: impl Fn(Iter<f32>) -> T) -> T {
        accessor(self.data.iter())
    }

    pub fn chain_zip_data<T>(
        &self,
        other: &Matrix,
        accessor: impl Fn(Zip<Iter<f32>, Iter<f32>>) -> T,
    ) -> T {
        accessor(self.data.iter().zip(other.data.iter()))
    }
}

impl Index<usize> for Matrix {
    type Output = [f32];

    fn index<'a>(&'a self, i: usize) -> &'a [f32] {
        let start = i * self.width;
        &self.data[start..start + self.width]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut [f32] {
        let start = i * self.width;
        &mut self.data[start..start + self.width]
    }
}

impl Display for Matrix {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                if let Err(e) = fmt.write_str(format!("| {:.1$}\t", self[y][x], 2).as_str()) {
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

pub struct MatrixRef {
    matrix: Rc<RefCell<Matrix>>,
}

impl MatrixRef {
    pub fn new(matrix: Matrix) -> Self {
        Self {
            matrix: Rc::new(RefCell::new(matrix)),
        }
    }

    pub fn get<'a>(&'a self) -> Ref<'a, Matrix> {
        self.matrix.borrow()
    }

    pub fn get_mut<'a>(&'a mut self) -> RefMut<'a, Matrix> {
        self.matrix.borrow_mut()
    }

    pub fn set(&mut self, matrix: Matrix) {
        self.get_mut().set(matrix);
    }
}

impl Display for MatrixRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mat = self.get();
        mat.fmt(f)
    }
}

impl Clone for MatrixRef {
    fn clone(&self) -> Self {
        Self {
            matrix: Rc::clone(&self.matrix),
        }
    }
}
