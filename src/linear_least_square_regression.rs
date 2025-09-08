use crate::matrix;
use crate::vector;

#[allow(non_snake_case)]
// Given a nxd matrix A, and a n sized vector b where n >= d.
// Return a d sized vector x such Ax - b is minimised.
// You can add and remove rows from A and b.
// It is the responsibility of the user to maintain invertibility of A.
pub struct LinearLeastSquareRegressionSolver {
    pub A: matrix::Matrix<f64>,
    pub b: vector::Vector<f64>,
    pub n: usize,
    pub d: usize,
    ATb: vector::Vector<f64>,
    ATAi: matrix::Matrix<f64>,
}

#[allow(non_snake_case)]
impl LinearLeastSquareRegressionSolver {
    pub fn init(A: matrix::Matrix<f64>, b: vector::Vector<f64>) -> Result<Self, ()> {
        assert!(
            A.num_rows == b.num_rows,
            "Number of rows don't match. A = {:?}, b = {:?}",
            A,
            b
        );
        let n = A.num_rows;
        let d = A.num_cols;

        let ATA = A.transpose() * A.clone();
        if ATA.det() == 0_f64 {
            return Err(());
        }

        let ATAi = ATA.inverse();
        // The operation Matrix * Vector is now defined to return a Vector
        let ATb = A.transpose() * b.clone();

        Ok(LinearLeastSquareRegressionSolver {
            A,
            b,
            n,
            d,
            ATb,
            ATAi,
        })
    }

    // Add a row of size d to the system.
    pub fn add(&mut self, alpha: vector::Vector<f64>, beta: f64) -> Result<(), ()> {
        assert!(
            alpha.num_rows == self.d,
            "Alpha vector has incorrect dimensions."
        );

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

    // Remove a row of size d from the system.
    pub fn remove(&mut self, alpha: vector::Vector<f64>, beta: f64) -> Result<(), ()> {
        assert!(
            alpha.num_rows == self.d,
            "Alpha vector has incorrect dimensions."
        );

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

    pub fn solve(&self) -> vector::Vector<f64> {
        self.ATAi.clone() * self.ATb.clone()
    }
}
