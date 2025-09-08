pub mod linear_least_square_regression;
pub mod matrix;
pub mod vector;

use crate::linear_least_square_regression::LinearLeastSquareRegressionSolver;
use crate::matrix::Matrix;
use rand::Rng; // Still needed for gen_range
use std::env;
use std::time::Instant;

#[allow(non_snake_case)]
fn main() {
    unsafe {
        env::set_var("RUST_BACKTRACE", "1");
    }

    // Define the dimensions for the large matrices
    let n = 10000;
    let d = 100;

    // --- 1. Generate Matrix A ---
    let start_gen_a = Instant::now();
    println!("Generating {}x{} matrix A...", n, d);
    let mut a_data = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..d).map(|_| rand::random::<f64>()).collect();
        a_data.push(row);
    }
    let A = Matrix::<f64>::new(a_data, n, d);
    println!("\tDone in {:.2?}.", start_gen_a.elapsed());

    // --- 2. Generate Vector b ---
    let start_gen_b = Instant::now();
    println!("Generating {}x1 vector b...", n);
    let mut b_data = Vec::with_capacity(n);
    for _ in 0..n {
        b_data.push(rand::random::<f64>());
    }
    let b = vector::Vector::<f64>::new(b_data);
    println!("\tDone in {:.2?}.", start_gen_b.elapsed());

    // --- 3. Initialize the solver ---
    let start_init = Instant::now();
    println!("Initializing solver...");
    let mut llsrs = LinearLeastSquareRegressionSolver::init(A.clone(), b.clone()).unwrap();
    println!("\tDone in {:.2?}.", start_init.elapsed());

    // --- 4. Perform initial solve ---
    let start_initial_solve = Instant::now();
    println!("Performing initial solve...");
    let _ = llsrs.solve();
    println!("\tDone in {:.2?}.", start_initial_solve.elapsed());

    // --- 5. Perform 1000 dynamic add/remove operations ---
    println!("\nStarting 1000 add/remove operations...");
    let num_operations = 1000;
    let mut dynamic_rows: Vec<(vector::Vector<f64>, f64)> = Vec::new();

    for i in 0..num_operations {
        let op_type: &str;
        let update_duration;

        let should_add = dynamic_rows.is_empty() || rand::random::<bool>();

        if should_add {
            op_type = "ADD";
            // CORRECTED LINE: Generate d rows, each with 1 column
            let mut new_alpha_data = Vec::with_capacity(d);
            for _ in 0..d {
                new_alpha_data.push(rand::random::<f64>());
            }
            let new_alpha = vector::Vector::new(new_alpha_data);
            let new_beta = rand::random::<f64>();

            let update_start = Instant::now();
            llsrs.add(new_alpha.clone(), new_beta).unwrap();
            update_duration = update_start.elapsed();

            dynamic_rows.push((new_alpha, new_beta));
        } else {
            op_type = "REMOVE";
            // We still need an RNG instance for gen_range
            let mut rng = rand::rng();
            let row_idx_to_remove = rng.random_range(0..dynamic_rows.len());
            let (alpha_to_remove, beta_to_remove) = dynamic_rows.swap_remove(row_idx_to_remove);

            let update_start = Instant::now();
            llsrs.remove(alpha_to_remove, beta_to_remove).unwrap();
            update_duration = update_start.elapsed();
        }

        // Solve after every operation
        let solve_start = Instant::now();
        let x = llsrs.solve();
        let solve_duration = solve_start.elapsed();

        // Prepare strings for formatted printing
        let op_str = format!("{}/{}", i + 1, num_operations);
        let type_str = format!("({})", op_type);
        let update_str = format!("{:?}", update_duration);
        let solve_str = format!("{:?}", solve_duration);
        let x_str = format!(
            "[{:.3}, {:.3}, {:.3}, ...]",
            x.mat[0][0], x.mat[1][0], x.mat[2][0]
        );

        // Log the result of this operation in an aligned format
        println!(
            "{:<12} {:<9} {:>15} {:>15}   {}",
            op_str, type_str, update_str, solve_str, x_str
        );
    }

    println!("\nAll operations complete.");
    println!(
        "NOTE: The 'Min Value' (||Ax-b||) is not calculated because it would require an O(n*d) operation at each step, defeating the purpose of the O(d^2) dynamic algorithm."
    );
}
