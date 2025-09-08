use std::cmp::PartialEq;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

// Trait alias for matrix element types
pub trait MatrixElement:
    Debug
    + Clone
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + PartialEq
    + std::ops::Neg
    + Identity
    + Zero
{
}

pub trait Identity {
    fn identity() -> Self;
}
pub trait Zero {
    fn zero() -> Self;
}

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub mat: Vec<Vec<T>>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl<T> Matrix<T>
where
    T: MatrixElement,
{
    // Zero matrix. Only defined for square matrices.
    pub fn zero(num_rows: usize, num_cols: usize) -> Self {
        Matrix {
            mat: vec![vec![T::zero(); num_cols]; num_rows],
            num_rows,
            num_cols,
        }
    }

    // Identity matrix. Only defined for square matrices.
    pub fn identity(size: usize) -> Self {
        let mut mat = vec![vec![T::zero(); size]; size];
        for i in 0..size {
            mat[i][i] = T::identity();
        }

        Matrix {
            mat,
            num_rows: size,
            num_cols: size,
        }
    }

    pub fn new(mat: Vec<Vec<T>>, num_rows: usize, num_cols: usize) -> Self {
        assert!(
            mat.len() == num_rows,
            "Number of rows doesn't match input. Expected {0} rows but received {1} rows.",
            num_rows,
            mat.len()
        );
        for (index, row) in mat.iter().enumerate() {
            assert!(
                row.len() == num_cols,
                "Number of columns doesn't match input. Expected {} columns but row {} has {} columns.",
                num_cols,
                index,
                row.len()
            );
        }

        Matrix {
            mat,
            num_rows,
            num_cols,
        }
    }

    // Takes O(n!) time. Only use when the matrices are small because its simple and no floating point division.
    fn laplacian_det(&self) -> T {
        if self.num_rows == 1 {
            return self.mat[0][0].clone();
        }
        if self.num_rows == 2 {
            return self.mat[0][0].clone() * self.mat[1][1].clone()
                - self.mat[0][1].clone() * self.mat[1][0].clone();
        }

        let mut det = T::zero();
        for col in 0..self.num_cols {
            // Build the minor matrix
            let mut minor = Vec::with_capacity(self.num_rows - 1);
            for i in 1..self.num_rows {
                let mut row = Vec::with_capacity(self.num_cols - 1);
                for j in 0..self.num_cols {
                    if j != col {
                        row.push(self.mat[i][j].clone());
                    }
                }
                minor.push(row);
            }
            let minor_matrix = Matrix::<T>::new(minor, self.num_rows - 1, self.num_cols - 1);
            let sign = if col % 2 == 0 {
                T::identity()
            } else {
                T::identity() * T::zero() - T::identity()
            };
            det = det + sign * self.mat[0][col].clone() * minor_matrix.det();
        }
        det
    }

    // Takes O(n^3) time using Gaussian elimination without pivoting
    fn gaussian_elimination_det(&self) -> T {
        let mut matrix = self.mat.clone();
        let n = self.num_rows;
        let mut sign = T::identity();

        for i in 0..n {
            // --- Pivoting for Correctness ---
            // Find the first row (from i downwards) with a non-zero pivot
            let mut pivot_row = i;
            while pivot_row < n && matrix[pivot_row][i] == T::zero() {
                pivot_row += 1;
            }

            // If we searched all rows and found no non-zero pivot,
            // the column is all zeros, so the determinant is zero.
            if pivot_row == n {
                return T::zero();
            }

            // Swap if we found a suitable pivot in a different row
            if pivot_row != i {
                matrix.swap(i, pivot_row);
                sign = -sign;
            }
            // --- End Pivoting ---

            // Elimination proceeds as before
            for k in i + 1..n {
                // The pivot matrix[i][i] is now guaranteed to be non-zero
                let factor = matrix[k][i] / matrix[i][i];
                for j in i..n {
                    matrix[k][j] = matrix[k][j] - factor * matrix[i][j];
                }
            }
        }

        let mut det = T::identity();
        for i in 0..n {
            det = det * matrix[i][i];
        }

        det * sign
    }

    // Uses laplacian method when size <= 4. Uses gaussian elimination otherwise which takes O(n^3) time.
    pub fn det(&self) -> T {
        assert!(
            self.num_rows == self.num_cols,
            "The matrix is not a square matrix, so the determinant is not defined. Matrix given = {:?}",
            self
        );
        if self.num_rows <= 4 {
            return self.laplacian_det();
        }
        self.gaussian_elimination_det()
    }

    /// Calculates the inverse of a square matrix using Gauss-Jordan elimination.
    ///
    /// The time complexity of this method is O(n³).
    ///
    /// # Panics
    ///
    /// This function will assert if:
    /// 1. The matrix is not square.
    /// 2. The matrix is singular (i.e., its determinant is zero) and thus not invertible.
    pub fn inverse(&self) -> Self {
        assert!(
            self.num_rows == self.num_cols,
            "The matrix is not a square matrix, so the inverse is not defined. Matrix given = {:?}",
            self
        );

        assert!(
            self.det() != T::zero(),
            "The determinant of the matrix is zero. Matrix = {:?}",
            self
        );

        let n = self.num_rows;
        let mut matrix = self.mat.clone(); // A mutable copy of our matrix
        let mut inv = Matrix::<T>::identity(n); // Starts as identity, will become the inverse

        // --- Phase 1: Forward Elimination (transforming to upper triangular) ---
        for i in 0..n {
            // 1. Pivoting: Find a non-zero pivot to avoid division by zero
            let mut pivot_row = i;
            while pivot_row < n && matrix[pivot_row][i] == T::zero() {
                pivot_row += 1;
            }

            if pivot_row == n {
                // No non-zero pivot found in this column, so the matrix is singular.
                panic!(
                    "Something went wrong. Determinant is non-zero but not finding pivot row. Maybe numeric instability with floats? Matrix = {:?}",
                    self
                );
            }

            // Swap the current row with the pivot row
            if pivot_row != i {
                matrix.swap(i, pivot_row);
                inv.mat.swap(i, pivot_row);
            }

            // 2. Normalization: Scale the pivot row so the pivot element is 1
            let pivot_val = matrix[i][i];
            for j in i..n {
                matrix[i][j] = matrix[i][j] / pivot_val;
            }
            for j in 0..n {
                inv.mat[i][j] = inv.mat[i][j] / pivot_val;
            }

            // 3. Elimination: Eliminate the elements below the pivot
            for k in i + 1..n {
                let factor = matrix[k][i];
                for j in i..n {
                    matrix[k][j] = matrix[k][j] - factor * matrix[i][j];
                }
                for j in 0..n {
                    inv.mat[k][j] = inv.mat[k][j] - factor * inv.mat[i][j];
                }
            }
        }

        // --- Phase 2: Backward Substitution (transforming to identity) ---
        // At this point, `matrix` is upper triangular with 1s on the diagonal.
        for i in (0..n).rev() {
            // Eliminate the elements above the pivot
            for k in 0..i {
                let factor = matrix[k][i];
                // No need to update the `matrix` copy, as its values are known (0s and 1s),
                // but we must update the `inv` matrix.
                for j in 0..n {
                    inv.mat[k][j] = inv.mat[k][j] - factor * inv.mat[i][j];
                }
            }
        }

        inv
    }
    // Performs O(logn) multiplication operations using recursive binary exponentiation.
    pub fn pow(&self, n: i32) -> Self {
        assert!(
            self.num_rows == self.num_cols,
            "The matrix is not a square matrix, so the power is not defined. Matrix given = {:?}",
            self
        );

        if n < 0 {
            todo!()
        }

        if n == 0 {
            return Matrix::identity(self.num_rows);
        }

        let mut result = self.clone().pow(n / 2);
        result = result.clone() * result;

        if n % 2 == 1 {
            result = result * self.clone();
        }
        result
    }

    pub fn transpose(&self) -> Matrix<T> {
        let mut transposed = Vec::with_capacity(self.num_cols);
        for col in 0..self.num_cols {
            let mut new_row = Vec::with_capacity(self.num_rows);
            for row in 0..self.num_rows {
                new_row.push(self.mat[row][col].clone());
            }
            transposed.push(new_row);
        }
        Matrix {
            mat: transposed,
            num_rows: self.num_cols,
            num_cols: self.num_rows,
        }
    }

    // Converts to T if Matrix is 1x1
    pub fn to_element(&self) -> T {
        assert!(self.num_cols == self.num_rows && self.num_rows == 1);
        self.mat[0][0]
    }
}

// Matrix + Matrix
impl<T> Add for Matrix<T>
where
    T: MatrixElement,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        assert!(
            self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols,
            "Matrices have different number of shape. LHS Matrix = {:?}, RHS matrix = {:?}.",
            self,
            rhs
        );

        let mut result = Matrix::<T>::zero(self.num_rows, self.num_cols);

        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                result.mat[i][j] = self.mat[i][j].clone() + rhs.mat[i][j].clone();
            }
        }
        result
    }
}

// Matrix - Matrix
impl<T> Sub for Matrix<T>
where
    T: MatrixElement,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        assert!(
            self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols,
            "Matrices have different number of shape. LHS Matrix = {:?}, RHS matrix = {:?}.",
            self,
            rhs
        );

        let mut result = Matrix::<T>::zero(self.num_rows, self.num_cols);

        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                result.mat[i][j] = self.mat[i][j].clone() - rhs.mat[i][j].clone();
            }
        }
        result
    }
}

// Matrix * Matrix
impl<T> Mul for Matrix<T>
where
    T: MatrixElement,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        assert!(
            self.num_cols == rhs.num_rows,
            "Number of columns in LHS Matrix is different from number of rows in RHS Matrix. LHS Matrix = {:?}, RHS matrix = {:?}.",
            self,
            rhs
        );

        let mut result = Matrix::<T>::zero(self.num_rows, rhs.num_cols);

        for i in 0..self.num_rows {
            for j in 0..rhs.num_cols {
                for k in 0..self.num_cols {
                    result.mat[i][j] += self.mat[i][k].clone() * rhs.mat[k][j].clone();
                }
            }
        }
        result
    }
}

// Matrix == Matrix
impl<T> PartialEq for Matrix<T>
where
    T: MatrixElement,
{
    fn eq(&self, rhs: &Self) -> bool {
        assert!(
            self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols,
            "Equality is not defined on matrices of different shapes. LHS Matrix = {:?}, RHS matrix = {:?}.",
            self,
            rhs
        );

        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                if self.mat[i][j] != rhs.mat[i][j] {
                    return false;
                }
            }
        }
        true
    }
}
// Matrix * scalar
impl<T> Mul<T> for Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                result.mat[i][j] = result.mat[i][j] * rhs;
            }
        }
        result
    }
}

// f64 specific implementation.
impl MatrixElement for f64 {}

impl Identity for f64 {
    fn identity() -> Self {
        1_f64
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0_f64
    }
}

// f64 * Matrix -- because generic implementation is not possible.
impl Mul<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn mul(self, rhs: Matrix<f64>) -> Self::Output {
        let mut result = rhs.clone();
        for i in 0..rhs.num_rows {
            for j in 0..rhs.num_cols {
                result.mat[i][j] = result.mat[i][j] * self;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Custom epsilon-based equality for f64
    const EPSILON: f64 = 1e-6;

    fn assert_approx_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "assertion failed: `(left == right)`\n  left: `{}`\n right: `{}`\n epsilon: `{}`",
            actual,
            expected,
            EPSILON
        );
    }

    #[test]
    fn test_determinant_1x1() {
        let matrix = Matrix::new(vec![vec![5.0]], 1, 1);
        assert_approx_eq(matrix.det(), 5.0);
    }

    #[test]
    fn test_determinant_2x2() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], 2, 2);
        // det = 1*4 - 2*3 = 4 - 6 = -2
        assert_approx_eq(matrix.det(), -2.0);
    }

    #[test]
    fn test_determinant_2x2_zero() {
        let matrix = Matrix::new(vec![vec![0.0, 1.0], vec![2.0, 0.0]], 2, 2);
        // det = 1*4 - 2*2 = 4 - 4 = 0
        assert_approx_eq(matrix.det(), -2.0);
    }

    #[test]
    fn test_determinant_3x3() {
        let matrix = Matrix::new(
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            3,
            3,
        );
        // Using Laplace expansion: det = 1*(5*9-6*8) - 2*(4*9-6*7) + 3*(4*8-5*7)
        // = 1*(45-48) - 2*(36-42) + 3*(32-35)
        // = 1*(-3) - 2*(-6) + 3*(-3)
        // = -3 + 12 - 9 = 0
        assert_approx_eq(matrix.det(), 0.0);
    }

    #[test]
    fn test_determinant_3x3_nonzero() {
        let matrix = Matrix::new(
            vec![
                vec![2.0, 1.0, 1.0],
                vec![1.0, 2.0, 1.0],
                vec![1.0, 1.0, 2.0],
            ],
            3,
            3,
        );
        // det = 2*(2*2-1*1) - 1*(1*2-1*1) + 1*(1*1-2*1)
        // = 2*(4-1) - 1*(2-1) + 1*(1-2)
        // = 2*3 - 1*1 + 1*(-1)
        // = 6 - 1 - 1 = 4
        assert_approx_eq(matrix.det(), 4.0);
    }

    #[test]
    fn test_determinant_4x4() {
        let matrix = Matrix::new(
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 4.0],
            ],
            4,
            4,
        );
        // Diagonal matrix: det = 1 * 2 * 3 * 4 = 24
        assert_approx_eq(matrix.det(), 24.0);
    }

    #[test]
    fn test_determinant_5x5_gaussian() {
        let matrix = Matrix::new(
            vec![
                vec![2.0, 1.0, 0.0, 0.0, 0.0],
                vec![1.0, 2.0, 1.0, 0.0, 0.0],
                vec![0.0, 1.0, 2.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0, 1.0, 2.0],
            ],
            5,
            5,
        );
        // This will use Gaussian elimination (size > 4)
        // The determinant should be positive (tridiagonal matrix with positive diagonal)
        let det = matrix.det();
        assert_approx_eq(det, 6.0);
    }

    #[test]
    fn test_determinant_identity() {
        let identity_3 = Matrix::<f64>::identity(3);
        assert_approx_eq(identity_3.det(), 1.0);

        let identity_5 = Matrix::<f64>::identity(5);
        assert_approx_eq(identity_5.det(), 1.0);
    }

    #[test]
    fn test_determinant_zero_matrix() {
        let zero_3 = Matrix::<f64>::zero(3, 3);
        assert_approx_eq(zero_3.det(), 0.0);

        let zero_4 = Matrix::<f64>::zero(4, 4);
        assert_approx_eq(zero_4.det(), 0.0);
    }

    #[test]
    fn test_determinant_upper_triangular() {
        let matrix = Matrix::new(
            vec![
                vec![2.0, 1.0, 3.0],
                vec![0.0, 3.0, 1.0],
                vec![0.0, 0.0, 4.0],
            ],
            3,
            3,
        );
        // Upper triangular: det = 2 * 3 * 4 = 24
        assert_approx_eq(matrix.det(), 24.0);
    }

    #[test]
    fn test_determinant_lower_triangular() {
        let matrix = Matrix::new(
            vec![
                vec![2.0, 0.0, 0.0],
                vec![1.0, 3.0, 0.0],
                vec![3.0, 1.0, 4.0],
            ],
            3,
            3,
        );
        // Lower triangular: det = 2 * 3 * 4 = 24
        assert_approx_eq(matrix.det(), 24.0);
    }

    #[test]
    #[should_panic(expected = "The matrix is not a square matrix")]
    fn test_determinant_non_square() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]], 3, 2);
        matrix.det();
    }

    // Helper function to compare two matrices for approximate equality
    fn assert_matrix_approx_eq(actual: &Matrix<f64>, expected: &Matrix<f64>) {
        assert_eq!(
            actual.num_rows, expected.num_rows,
            "Matrix row counts differ"
        );
        assert_eq!(
            actual.num_cols, expected.num_cols,
            "Matrix column counts differ"
        );

        for i in 0..actual.num_rows {
            for j in 0..actual.num_cols {
                assert!(
                    (actual.mat[i][j] - expected.mat[i][j]).abs() < EPSILON,
                    "assertion failed at ({}, {}): `(left == right)`\n  left: `{}`\n right: `{}`",
                    i,
                    j,
                    actual.mat[i][j],
                    expected.mat[i][j]
                );
            }
        }
    }

    #[test]
    fn test_inverse_2x2() {
        let matrix = Matrix::new(vec![vec![4.0, 7.0], vec![2.0, 6.0]], 2, 2);
        // Inverse of [[a, b], [c, d]] is 1/(ad-bc) * [[d, -b], [-c, a]]
        // det = 4*6 - 7*2 = 24 - 14 = 10
        // inv = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        let expected_inverse = Matrix::new(vec![vec![0.6, -0.7], vec![-0.2, 0.4]], 2, 2);
        let actual_inverse = matrix.inverse();
        assert_matrix_approx_eq(&actual_inverse, &expected_inverse);
    }

    #[test]
    fn test_inverse_3x3() {
        let matrix = Matrix::new(
            vec![
                vec![2.0, -1.0, 0.0],
                vec![-1.0, 2.0, -1.0],
                vec![0.0, -1.0, 2.0],
            ],
            3,
            3,
        );
        let expected_inverse = Matrix::new(
            vec![
                vec![0.75, 0.5, 0.25],
                vec![0.5, 1.0, 0.5],
                vec![0.25, 0.5, 0.75],
            ],
            3,
            3,
        );
        let actual_inverse = matrix.inverse();
        assert_matrix_approx_eq(&actual_inverse, &expected_inverse);
    }

    #[test]
    #[should_panic(expected = "The determinant of the matrix is zero.")]
    fn test_inverse_of_singular_matrix() {
        // This matrix has a determinant of 0
        let matrix = Matrix::new(
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            3,
            3,
        );
        // This should panic
        matrix.inverse();
    }

    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix::new(
            vec![
                vec![2.0, 1.0, 1.0],
                vec![1.0, 3.0, -1.0],
                vec![1.0, 1.0, 4.0],
            ],
            3,
            3,
        );
        let inverse_matrix = matrix.clone().inverse();
        let result = matrix * inverse_matrix;
        let identity = Matrix::<f64>::identity(3);

        // Check if M * M⁻¹ ≈ I
        assert_matrix_approx_eq(&result, &identity);
    }

    #[test]
    fn test_determinant_of_matrix_times_determinant_of_inverse() {
        let matrix = Matrix::new(
            vec![
                vec![5.0, 2.0, -1.0],
                vec![1.0, 7.0, 3.0],
                vec![2.0, 4.0, 6.0],
            ],
            3,
            3,
        );

        // An important property of determinants is det(M⁻¹) = 1 / det(M)
        // Therefore, det(M) * det(M⁻¹) should equal 1.

        let det_m = matrix.det();
        let inverse_matrix = matrix.inverse();
        let det_m_inv = inverse_matrix.det();

        let result = det_m * det_m_inv;

        // Check if det(M) * det(M⁻¹) ≈ 1.0
        assert_approx_eq(result, 1.0);
    }
}
