use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Display,
    iter::Zip,
    ops::{DerefMut, Index, IndexMut},
    rc::Rc,
    slice::Iter,
};

use rand::{distributions::Normal, Rng};

use crate::operation::Operation;

/* ---------------------------------------------------------------------- */

#[derive(Clone)]
pub struct MatrixData {
    height: usize,
    width: usize,
    data: Vec<f32>,
}

impl MatrixData {
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

    pub fn set(&mut self, other: MatrixData) {
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
        other: &MatrixData,
        accessor: impl Fn(Zip<Iter<f32>, Iter<f32>>) -> T,
    ) -> T {
        accessor(self.data.iter().zip(other.data.iter()))
    }
}

impl Index<usize> for MatrixData {
    type Output = [f32];

    fn index<'a>(&'a self, i: usize) -> &'a [f32] {
        let start = i * self.width;
        &self.data[start..start + self.width]
    }
}

impl IndexMut<usize> for MatrixData {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut [f32] {
        let start = i * self.width;
        &mut self.data[start..start + self.width]
    }
}

impl Display for MatrixData {
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

/* ---------------------------------------------------------------------- */

pub struct Matrix {
    data: Rc<RefCell<MatrixData>>,
    grad: Rc<RefCell<MatrixData>>,

    operation: Option<Box<dyn Operation>>,
}

impl Matrix {
    pub fn new(data: MatrixData) -> Self {
        let height = data.height;
        let width = data.width;

        Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(MatrixData::zeros(height, width))),

            operation: None,
        }
    }

    pub fn new_output<O: Operation + 'static>(data: MatrixData, operation: O) -> Self {
        let height = data.height;
        let width = data.width;

        Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(MatrixData::zeros(height, width))),

            operation: Some(Box::new(operation)),
        }
    }

    pub fn data<'a>(&'a self) -> Ref<'a, MatrixData> {
        self.data.as_ref().borrow()
    }

    pub fn data_mut<'a>(&'a mut self) -> RefMut<'a, MatrixData> {
        self.data.as_ref().borrow_mut()
    }

    pub fn grad<'a>(&'a self) -> Ref<'a, MatrixData> {
        self.grad.as_ref().borrow()
    }

    pub fn grad_mut<'a>(&'a mut self) -> RefMut<'a, MatrixData> {
        self.grad.as_ref().borrow_mut()
    }

    pub fn back(&mut self) {
        let width = self.data.borrow().width;
        let height = self.data.borrow().height;

        let delta = MatrixData {
            width,
            height,
            data: (0..width * height).map(|_| 1.0).collect(),
        };

        self.back_delta(delta);
    }

    pub fn back_delta(&mut self, delta: MatrixData) {
        let mut grad_guard = self.grad_mut();
        let grad_ref = grad_guard.deref_mut();

        debug_assert_eq!(delta.width, grad_ref.width);
        debug_assert_eq!(delta.height, grad_ref.height);

        for delta_val in delta.data {
            for grad_val in grad_ref.data.iter_mut() {
                *grad_val += delta_val;
            }
        }

        if let Some(operation) = &self.operation {
            operation.back_grad(delta);
        }
    }
}
