use cs8803_dynamic_algebraic_algorithms_implementation::Matrix;
use cs8803_dynamic_algebraic_algorithms_implementation::week1::dag_all_pairs_reachability::DagReachabilitySolver;

fn main() {
    let adj = Matrix::<i64>::new(vec![vec![0, 1, 1], vec![0, 0, 1], vec![0, 0, 0]], 3, 3);
    let mut dag_reachability_solver = DagReachabilitySolver::new(adj);

    println!("{:?}", dag_reachability_solver.get_reachbility_matrix());

    dag_reachability_solver.remove(1, 2);

    println!("{:?}", dag_reachability_solver.get_reachbility_matrix());

    dag_reachability_solver.add(0, 1);

    println!("{:?}", dag_reachability_solver.get_reachbility_matrix());
}
