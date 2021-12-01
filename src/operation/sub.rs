use std::ops::Sub;

use crate::operation::Operation;

use super::OperationRef;

impl Sub for OperationRef {
    type Output = OperationRef;

    fn sub(self, rhs: OperationRef) -> Self::Output {
        self + rhs.times(-1.0)
    }
}
