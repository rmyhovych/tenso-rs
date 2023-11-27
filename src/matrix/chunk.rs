const CHUNK_DIMENSION: usize = 4;
const CHUNK_WIDTH: usize = 1 << CHUNK_DIMENSION;

const CHUNK_MASK: usize = CHUNK_WIDTH - 1;

pub struct MatrixChunk {
    data: [f32; CHUNK_WIDTH * CHUNK_WIDTH],
}

impl MatrixChunk {
    pub fn new() -> Self {
        Self {
            data: [0.0; CHUNK_WIDTH * CHUNK_WIDTH],
        }
    }

    pub fn for_each<TFuncType: FnMut([usize; 2], f32)>(&self, func: &mut TFuncType) {
        let mut index = 0;
        for y in 0..CHUNK_WIDTH {
            for x in 0..CHUNK_WIDTH {
                func([y, x], self.data[index]);
                index += 1;
            }
        }
    }

    pub fn for_each_mut<TFuncType: FnMut([usize; 2], f32) -> f32>(&mut self, func: &mut TFuncType) {
        let mut index = 0;
        for y in 0..CHUNK_WIDTH {
            for x in 0..CHUNK_WIDTH {
                self.data[index] = func([y, x], self.data[index]);
                index += 1;
            }
        }
    }

    pub fn set(&mut self, coord: [usize; 2], value: f32) {
        self.data[Self::coord_to_index(coord)] = value;
    }

    pub fn get(&self, coord: [usize; 2]) -> &f32 {
        &self.data[Self::coord_to_index(coord)]
    }

    pub fn get_mut(&mut self, coord: [usize; 2]) -> &mut f32 {
        &mut self.data[Self::coord_to_index(coord)]
    }

    pub fn set_unbounded(&mut self, global_coord: [usize; 2], value: f32) {
        self.set(Self::bound_coord(global_coord), value);
    }

    pub fn get_unbounded(&self, global_coord: [usize; 2]) -> &f32 {
        self.get(Self::bound_coord(global_coord))
    }

    pub fn get_unbounded_mut(&mut self, global_coord: [usize; 2]) -> &mut f32 {
        self.get_mut(Self::bound_coord(global_coord))
    }

    /* ------------------------------------------------- */

    pub fn unary_assign_operation<TFuncType>(&mut self, func: &TFuncType)
    where
        TFuncType: Fn(f32) -> f32,
    {
        for i in 0..self.data.len() {
            self.data[i] = func(self.data[i]);
        }
    }

    pub fn unary_operation<TFuncType>(&self, func: &TFuncType) -> Self
    where
        TFuncType: Fn(f32) -> f32,
    {
        let mut result = Self::new();
        result.unary_assign_operation(func);
        result
    }

    pub fn binary_assign_operation<TFuncType>(&mut self, other: &Self, func: &TFuncType)
    where
        TFuncType: Fn(f32, f32) -> f32,
    {
        for i in 0..self.data.len() {
            self.data[i] = func(self.data[i], other.data[i]);
        }
    }

    pub fn binary_operation<TFuncType>(&self, other: &Self, func: &TFuncType) -> Self
    where
        TFuncType: Fn(f32, f32) -> f32,
    {
        let mut result = Self::new();
        result.binary_assign_operation(other, func);
        result
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new();
        for y in 0..CHUNK_WIDTH {
            for x in 0..CHUNK_WIDTH {
                result.set([y, x], *self.get([x, y]));
            }
        }
        result
    }

    pub fn matmul(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for y in 0..CHUNK_WIDTH {
            for x in 0..CHUNK_WIDTH {
                let mut value: f32 = 0.0;
                for i in 0..CHUNK_WIDTH {
                    value += self.get([y, i]) * other.get([i, x]);
                }

                result.set([y, x], value);
            }
        }
        result
    }

    /* ------------------------------------------------- */

    pub fn get_chunk_index(data_index: [usize; 2]) -> [usize; 2] {
        [
            data_index[0] >> CHUNK_DIMENSION,
            data_index[1] >> CHUNK_DIMENSION,
        ]
    }

    pub fn get_chunk_size(data_size: [usize; 2]) -> [usize; 2] {
        let index = Self::get_chunk_index([data_size[0] - 1, data_size[1] - 1]);
        [index[0] + 1, index[1] + 1]
    }

    pub fn get_element_size(chunk_size: [usize; 2]) -> [usize; 2] {
        [CHUNK_WIDTH * chunk_size[0], CHUNK_WIDTH * chunk_size[1]]
    }

    /* ------------------------------------------------- */

    #[inline]
    fn bound_coord(coord: [usize; 2]) -> [usize; 2] {
        [coord[0] & CHUNK_MASK, coord[1] & CHUNK_MASK]
    }

    #[inline]
    fn coord_to_index(coord: [usize; 2]) -> usize {
        (coord[0] << CHUNK_DIMENSION) + coord[1]
    }
}
