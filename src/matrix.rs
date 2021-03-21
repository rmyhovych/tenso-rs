use std::{cell::RefCell, fmt::Display, iter::Zip, ops::Deref, rc::Rc, slice::Iter};

use rand::{distributions::Normal, Rng};

#[derive(Clone)]
pub struct Matrix {
    mat: Rc<RefCell<MatrixBase>>,
}

impl Matrix {
    pub fn new(height: usize, width: usize, data: Vec<f32>) -> Self {
        Self::from_base(MatrixBase {
            height,
            width,
            data,
        })
    }

    pub fn zeros(height: usize, width: usize) -> Self {
        Self::from_base(MatrixBase::zeros(height, width))
    }

    pub fn from_const(height: usize, width: usize, value: f32) -> Self {
        Self::from_base(MatrixBase::from_const(height, width, value))
    }

    pub fn randn(height: usize, width: usize, mean: f64, std: f64) -> Self {
        Self::from_base(MatrixBase::randn(height, width, mean, std))
    }

    fn from_base(mat: MatrixBase) -> Self {
        Self {
            mat: Rc::new(RefCell::new(mat)),
        }
    }

    /*------------------------------------------------------*/

    pub fn clear(&mut self) {
        self.mat
            .as_ref()
            .borrow_mut()
            .data
            .iter_mut()
            .for_each(|v| *v = 0.0);
    }

    pub fn set(&mut self, other: &Matrix) {
        let other_ref = other.mat.borrow();
        self.mat.as_ref().borrow_mut().set(other_ref.deref());
    }

    pub fn get_height(&self) -> usize {
        self.mat.borrow().height
    }

    pub fn get_width(&self) -> usize {
        self.mat.borrow().width
    }

    pub fn get_value(&self, y: usize, x: usize) -> f32 {
        self.mat.borrow().get_value(y, x)
    }

    pub fn set_value(&mut self, val: f32, y: usize, x: usize) {
        self.mat.as_ref().borrow_mut().set_value(val, y, x);
    }

    pub fn chain_data<T>(&self, accessor: impl Fn(Iter<f32>) -> T) -> T {
        accessor(self.mat.borrow().data.iter())
    }

    pub fn chain_zip_data<T>(
        &self,
        other: &Matrix,
        accessor: impl Fn(Zip<Iter<f32>, Iter<f32>>) -> T,
    ) -> T {
        let ref0 = self.mat.borrow();
        let ref1 = other.mat.borrow();
        let iter0 = ref0.data.iter();
        let iter1 = ref1.data.iter();

        accessor(iter0.zip(iter1))
    }
}

impl Display for Matrix {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.mat.borrow().fmt(fmt)
    }
}

/*------------------------------------------------------------------------------------------------*/

struct MatrixBase {
    height: usize,
    width: usize,
    data: Vec<f32>,
}

impl MatrixBase {
    fn zeros(height: usize, width: usize) -> Self {
        Self::from_const(height, width, 0.0)
    }

    fn from_const(height: usize, width: usize, value: f32) -> Self {
        Self {
            height,
            width,
            data: (0..height * width).map(|_| value).collect(),
        }
    }

    fn randn(height: usize, width: usize, mean: f64, std: f64) -> Self {
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

    fn set(&mut self, other: &MatrixBase) {
        self.width = other.width;
        self.height = other.height;
        self.data = other.data.clone();
    }

    fn get_value(&self, y: usize, x: usize) -> f32 {
        self.data[y * self.width + x]
    }

    fn set_value(&mut self, val: f32, y: usize, x: usize) {
        self.data[y * self.width + x] = val;
    }
}

impl Display for MatrixBase {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                if let Err(e) =
                    fmt.write_str(format!("| {:.1$}\t", self.get_value(y, x), 2).as_str())
                {
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
