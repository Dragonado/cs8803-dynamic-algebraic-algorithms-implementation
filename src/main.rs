pub mod linear_least_square_regression;
pub mod matrix;

use crate::linear_least_square_regression::LinearLeastSquareRegressionSolver;
use crate::matrix::Matrix;

// pub mod linear_least_square_regression;
// pub mod matrix;
#[allow(non_snake_case)]
fn main() {
    let A = Matrix::<f64>::identity(3) * 2_f64;
    let b = Matrix::<f64>::new(vec![vec![1_f64]; 3], 3_usize, 1_usize);
    let llsrs = LinearLeastSquareRegressionSolver::init(A, b);
    // let items: Vec<Vec<f32>> = vec![vec![1.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]];
    // let m = matrix::Matrix::<f32>::new(items, 2, 3);
    println!("{:?}", llsrs.solve());
}
