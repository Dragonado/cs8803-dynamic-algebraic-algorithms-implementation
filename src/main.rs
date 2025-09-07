pub mod matrix;

fn main() {
    // let items: Vec<Vec<f32>> = vec![vec![1.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]];
    // let m = matrix::Matrix::<f32>::new(items, 2, 3);
    println!(
        "{:?}",
        (matrix::Matrix::<f32>::identity(5) + matrix::Matrix::<f32>::identity(5)).det()
    );
}
