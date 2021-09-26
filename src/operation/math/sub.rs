use std::ops::Sub;

use crate::operation::Operation;

impl Sub for Operation {
    type Output = Operation;

    fn sub(self, rhs: Operation) -> Self::Output {
        self + rhs.times(-1.0)
    }
}
