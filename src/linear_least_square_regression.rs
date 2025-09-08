use crate::matrix;

#[allow(non_snake_case)]
// Given a nxd matrix A, and a n sized vector b where n >= d.
// Return a d sized vector x such Ax - b is minimised.
// You can add and remove rows from A and b.
// It is the responsibility of the user to maintain invertibility of A.
pub struct LinearLeastSquareRegressionSolver {
    pub A: matrix::Matrix<f64>,
    pub b: matrix::Matrix<f64>, // TODO: Replace with vector when implmentation is ready.
    pub n: usize,
    pub d: usize,
    ATb: matrix::Matrix<f64>, // TODO: Replace with vector when implementation is ready.
    ATAi: matrix::Matrix<f64>,
}

#[allow(non_snake_case)]
impl LinearLeastSquareRegressionSolver {
    pub fn init(A: matrix::Matrix<f64>, b: matrix::Matrix<f64>) -> Self {
        assert!(
            A.num_rows == b.num_rows,
            "Number of rows don't match. A = {:?}, b = {:?}",
            A,
            b
        );
        assert!(b.num_cols == 1, "b is not a vector. b = {:?}", b);
        let n = A.num_rows;
        let d = A.num_cols;

        let ATAi = (A.transpose() * A.clone()).inverse();
        let ATb = A.transpose() * b.clone();

        LinearLeastSquareRegressionSolver {
            A,
            b,
            n,
            d,
            ATb,
            ATAi,
        }
    }
    // Add a row of size d and remove the corresponding element in b.
    // Its the responsibility of the user to maintain invertibility of A.
    pub fn add(&mut self, alpha: matrix::Matrix<f64>, beta: f64) -> Result<(), ()> {
        assert!(alpha.num_cols == self.d);
        assert!(alpha.num_rows == 1);

        let den: f64 = 1_f64 + (alpha.transpose() * self.ATAi.clone() * alpha.clone()).to_element();

        if den == 0.0 {
            Err(())
        } else {
            self.ATAi = self.ATAi.clone()
                - (self.ATAi.clone() * alpha.clone())
                    * (alpha.transpose() * self.ATAi.clone())
                    * (1_f64 / den);
            self.ATb = self.ATb.clone() + alpha * beta;
            Ok(())
        }
    }

    // Remove a row of size d and remove the corresponding element in b.
    // Its the responsibility of the user to remove an row that actually exists.
    // Expect garbage if youre removing something that doesnt exist.
    pub fn remove(&mut self, alpha: matrix::Matrix<f64>, beta: f64) -> Result<(), ()> {
        assert!(alpha.num_cols == self.d);
        assert!(alpha.num_rows == 1);

        let den: f64 = 1_f64 - (alpha.transpose() * self.ATAi.clone() * alpha.clone()).to_element();

        if den == 0.0 {
            Err(())
        } else {
            self.ATAi = self.ATAi.clone()
                + (self.ATAi.clone() * alpha.clone())
                    * (alpha.transpose() * self.ATAi.clone())
                    * (1_f64 / den);
            self.ATb = self.ATb.clone() - alpha * beta;
            Ok(())
        }
    }

    pub fn solve(&self) -> matrix::Matrix<f64> {
        self.ATAi.clone() * self.ATb.clone()
    }
}
