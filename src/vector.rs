use crate::matrix::{Matrix, MatrixElement};
use std::convert::TryFrom;
use std::ops::Add;
use std::ops::Deref;
use std::ops::Mul;
use std::ops::Sub;

// A type-safe wrapper for an n x 1 matrix to represent a column vector.
#[derive(Debug, Clone)]
pub struct Vector<T: MatrixElement>(pub Matrix<T>); // The Vector is a wrapper around a Matrix

// fallback to Matrix implementation if not defined in Vector.
impl<T: MatrixElement> Deref for Vector<T> {
    type Target = Matrix<T>;

    fn deref(&self) -> &Self::Target {
        // Returns a reference to the inner Matrix, enabling method access.
        &self.0
    }
}

// Define generic Vector implementations.
impl<T: MatrixElement> Vector<T> {
    /// Creates a new column vector from a flat list of elements.
    /// This constructor ensures the underlying matrix is always n x 1.
    pub fn new(elements: Vec<T>) -> Self {
        let n = elements.len();

        assert!(n != 0, "0 dimensional vector is not defined.");
        // Internally create the n x 1 matrix data structure
        let mat_data: Vec<Vec<T>> = elements.into_iter().map(|elem| vec![elem]).collect();
        Vector(Matrix::new(mat_data, n, 1))
    }

    /// Calculates the dot product of two vectors.
    /// Panics if the vectors do not have the same length.
    pub fn dot(&self, rhs: &Self) -> T {
        assert_eq!(
            self.num_rows, rhs.num_rows,
            "Vectors must have the same length for dot product."
        );

        let mut sum = T::zero();
        for i in 0..self.num_rows {
            sum += self.mat[i][0] * rhs.mat[i][0];
        }
        sum
    }
}

// Matrix * Vector  -- returns a vector
impl<T: crate::matrix::MatrixElement> Mul<Vector<T>> for Matrix<T> {
    type Output = Vector<T>;
    fn mul(self, rhs: Vector<T>) -> Self::Output {
        // Reuse the existing Matrix * Matrix multiplication on the vector's inner matrix (rhs.0)
        let result_matrix = self * rhs.0;
        Vector(result_matrix)
    }
}

// Vector * Matrix  -- returns a matrix
impl<T: crate::matrix::MatrixElement> Mul<Matrix<T>> for Vector<T> {
    type Output = Matrix<T>;

    /// Defines the outer product for a Vector * Matrix (where Matrix is a row vector).
    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        // Assert that the right-hand side is a row vector (1 x m)
        assert_eq!(
            rhs.num_rows, 1,
            "Outer product requires the right-hand matrix to be a row vector (1xM)."
        );

        // Use the existing Matrix * Matrix multiplication.
        // `self` is dereferenced to its inner Matrix automatically.
        self.0 * rhs
    }
}

// Convert Matrix into Vector
impl<T: MatrixElement> TryFrom<Matrix<T>> for Vector<T> {
    // We'll use a String for a descriptive error message.
    type Error = String;

    fn try_from(matrix: Matrix<T>) -> Result<Self, Self::Error> {
        // Check if the matrix is a column vector.
        if matrix.num_cols == 1 {
            // If it is, wrap it in the Vector struct and return Ok.
            Ok(Vector(matrix))
        } else {
            // If not, return an error with a descriptive message.
            Err(format!(
                "Matrix must have exactly one column to be a Vector, but it has {}",
                matrix.num_cols
            ))
        }
    }
}

// Vector + Vector
impl<T: MatrixElement> Add<Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    /// Defines Vector + Vector addition.
    /// Panics if the vectors do not have the same length.
    fn add(self, rhs: Vector<T>) -> Self::Output {
        // Assert that the lengths are the same
        assert_eq!(
            self.num_rows, rhs.num_rows,
            "Vectors must have the same length for addition."
        );

        // Reuse the existing Matrix + Matrix addition on the inner matrices.
        Vector(self.0 + rhs.0)
    }
}

// Vector - Vector
impl<T: MatrixElement> Sub<Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    /// Defines Vector - Vector subtraction.
    /// Panics if the vectors do not have the same length.
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        // Assert that the lengths are the same
        assert_eq!(
            self.num_rows, rhs.num_rows,
            "Vectors must have the same length for subtraction."
        );

        // Reuse the existing Matrix - Matrix subtraction on the inner matrices.
        Vector(self.0 - rhs.0)
    }
}

// Vector * scalar
impl<T: crate::matrix::MatrixElement> Mul<T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Self::Output {
        // Reuses the generic Matrix<T> * T multiplication on the inner matrix
        let result_matrix = self.0 * rhs;

        // Wrap the result back into a Vector
        Vector(result_matrix)
    }
}

// f64 * Vector -- because generic implementation not possible.
impl Mul<Vector<f64>> for f64 {
    type Output = Vector<f64>;

    fn mul(self, rhs: Vector<f64>) -> Self::Output {
        // Reuse the logic from the Vector * f64 implementation
        rhs * self
    }
}

// L2 norm of a f64 Vector.
impl Vector<f64> {
    /// Calculates the L2 norm (Euclidean length) of the vector.
    pub fn l2_norm(&self) -> f64 {
        self.dot(&self).sqrt()
    }
}
