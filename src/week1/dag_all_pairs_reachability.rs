use crate::{matrix, prime_field::PrimeFieldElement};

const NUM_TRIES: usize = 100;

// Helper function to create zero matrix with specific prime
fn zero_matrix_with_prime(
    rows: usize,
    cols: usize,
    prime: u64,
) -> matrix::Matrix<PrimeFieldElement> {
    let mut mat = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push(PrimeFieldElement::new(0, prime));
        }
        mat.push(row);
    }
    matrix::Matrix::new(mat, rows, cols)
}

pub struct DagReachabilitySolver {
    pub adj: matrix::Matrix<i64>,
    reachable: Vec<matrix::Matrix<PrimeFieldElement>>, // replace with a array of size NUM_TRIES and use Matrix<Group>
    prime: u64,                                        // Store the prime for this solver
}

impl DagReachabilitySolver {
    // All initial edges must be from u -> v where u < v. This ensures a DAG structure.
    pub fn new(adj: matrix::Matrix<i64>, prime: u64) -> Self {
        assert!(
            adj.num_cols == adj.num_rows,
            "Matrix is not square. Adj = {:#?}",
            adj
        );

        for i in 0..adj.num_rows {
            for j in 0..(i + 1) {
                assert!(
                    adj.mat[i][j] == 0,
                    "There is a back edge (may or may not be DAG). adj = {:#?}",
                    adj
                );
            }
        }

        let mut reachable: Vec<matrix::Matrix<PrimeFieldElement>> = Vec::with_capacity(NUM_TRIES);
        for _ in 0..NUM_TRIES {
            reachable.push(zero_matrix_with_prime(adj.num_rows, adj.num_cols, prime));
        }
        DagReachabilitySolver {
            adj,
            reachable,
            prime,
        }
    }

    pub fn add(&mut self, u: usize, v: usize) {
        assert!(
            u < self.adj.num_rows,
            "Invalid vertex. u = {:#?}, v = {:#?}",
            u,
            v
        );
        assert!(
            v < self.adj.num_rows,
            "Invalid vertex. u = {:#?}, v = {:#?}",
            u,
            v
        );
        assert!(
            self.adj.mat[u][v] == 0,
            "There already exists an edge from u to v. u, v = {:#?}, {:#?}. Adj = {:#?}",
            u,
            v,
            self.adj
        );

        self.adj.mat[u][v] = 1;
    }

    pub fn remove(&mut self, u: usize, v: usize) {
        assert!(
            u < self.adj.num_rows,
            "Invalid vertex. u = {:#?}, v = {:#?}",
            u,
            v
        );
        assert!(
            v < self.adj.num_rows,
            "Invalid vertex. u = {:#?}, v = {:#?}",
            u,
            v
        );
        assert!(
            self.adj.mat[u][v] == 1,
            "No edge from u to v. u = {:#?}, v = {:#?}, adj = {:#?}",
            u,
            v,
            self.adj
        );

        self.adj.mat[u][v] = 0;
    }

    pub fn get_reachbility_matrix(&self) -> Vec<matrix::Matrix<PrimeFieldElement>> {
        self.reachable.clone()
    }
}
