use tinyopt::backend::{DenseMatrix, DenseVector, Dim, Matrix, Vector, zeros_matrix, zeros_vector};
use tinyopt::block_sparse::BlockSparseMat;
use tinyopt::hessian::HessianStorage;
use tinyopt::sparse::SparseMatrix;
use tinyopt::{
    Cost, Options, Params, Static, block_sparse_types, optimize_parallel,
    optimize_parallel_block_sparse, optimize_parallel_sparse,
};

const FACTOR_COUNT: usize = 64;

block_sparse_types!(ScalarBlock, [1]);

fn scalar_target(index: usize) -> f64 {
    1.0 + ((index % 8) as f64 - 3.5) * 0.1
}

fn scalar_cost(index: usize, x: f64) -> (f64, f64) {
    let residual = x - scalar_target(index);
    (0.5 * residual * residual, residual)
}

fn quadratic_factor<D: Dim>(
    index: usize,
    x: &[f64],
    gradient: Option<&mut Vector<f64, D>>,
    hessian: Option<&mut Matrix<f64, D>>,
) -> Cost {
    let coordinate = index % 2;
    let offset = ((index / 2 % 8) as f64 - 3.5) * 0.1;
    let target = if coordinate == 0 { 1.0 } else { -2.0 } + offset;
    let residual = x[coordinate] - target;

    if let Some(gradient) = gradient {
        gradient.set_at(coordinate, gradient.at(coordinate) + residual);
    }
    if let Some(hessian) = hessian {
        hessian.set_elem(
            coordinate,
            coordinate,
            hessian.elem(coordinate, coordinate) + 1.0,
        );
    }
    Cost::with_num_residuals(0.5 * residual * residual, 1)
}

fn total_cost<D: Dim>(x: &[f64]) -> f64 {
    (0..FACTOR_COUNT)
        .map(|index| quadratic_factor::<D>(index, x, None, None).cost)
        .sum()
}

fn total_gradient<D: Dim>(x: &[f64]) -> Vector<f64, D> {
    let mut gradient = zeros_vector::<f64, D>(x.len());
    for index in 0..FACTOR_COUNT {
        quadratic_factor::<D>(index, x, Some(&mut gradient), None);
    }
    gradient
}

fn check_derivatives<D: Dim>() {
    let x = [0.35, -0.75];
    let mut gradient = zeros_vector::<f64, D>(x.len());
    let mut hessian = zeros_matrix::<f64, D>(x.len(), x.len());
    for index in 0..FACTOR_COUNT {
        quadratic_factor::<D>(index, &x, Some(&mut gradient), Some(&mut hessian));
    }

    let epsilon = 1e-5;
    for coordinate in 0..x.len() {
        let mut plus = x;
        let mut minus = x;
        plus[coordinate] += epsilon;
        minus[coordinate] -= epsilon;
        let numeric_gradient = (total_cost::<D>(&plus) - total_cost::<D>(&minus)) / (2.0 * epsilon);
        assert!(
            (gradient.at(coordinate) - numeric_gradient).abs() < 1e-7,
            "gradient[{coordinate}]={} vs finite difference {numeric_gradient}",
            gradient.at(coordinate)
        );

        let gradient_plus = total_gradient::<D>(&plus);
        let gradient_minus = total_gradient::<D>(&minus);
        for row in 0..x.len() {
            let numeric_hessian =
                (gradient_plus.at(row) - gradient_minus.at(row)) / (2.0 * epsilon);
            assert!(
                (hessian.elem(row, coordinate) - numeric_hessian).abs() < 1e-7,
                "hessian[{row}, {coordinate}]={} vs finite difference {numeric_hessian}",
                hessian.elem(row, coordinate)
            );
        }
    }
}

fn optimize_problem<P>(mut params: P) -> P
where
    P: Params<Scalar = f64>,
    P::Dim: Dim,
    Matrix<f64, P::Dim>: HessianStorage<f64>,
{
    let summary = optimize_parallel(
        &mut params,
        FACTOR_COUNT,
        quadratic_factor::<P::Dim>,
        &Options::default(),
    );
    assert!(summary.converged(), "{:?}", summary.stop_reason);
    params
}

fn check_scalar_hessian(
    gradient: f64,
    hessian: f64,
    x: f64,
    cost: impl Fn(f64) -> f64,
    gradient_at: impl Fn(f64) -> f64,
) {
    let epsilon = 1e-5;
    let numeric_gradient = (cost(x + epsilon) - cost(x - epsilon)) / (2.0 * epsilon);
    let numeric_hessian = (gradient_at(x + epsilon) - gradient_at(x - epsilon)) / (2.0 * epsilon);
    assert!((gradient - numeric_gradient).abs() < 1e-7);
    assert!((hessian - numeric_hessian).abs() < 1e-5);
}

#[test]
fn factor_derivatives_match_finite_differences() {
    check_derivatives::<Static<2>>();
    check_derivatives::<tinyopt::Dynamic>();
}

#[test]
fn sparse_and_block_sparse_factor_derivatives_match_finite_differences() {
    let x = 0.35;
    let cost = |value: f64| {
        (0..FACTOR_COUNT)
            .map(|index| scalar_cost(index, value).0)
            .sum::<f64>()
    };

    let mut sparse_gradient = zeros_vector::<f64, tinyopt::Dynamic>(1);
    let mut sparse_hessian = SparseMatrix::new(1, 1);
    for index in 0..FACTOR_COUNT {
        let (_, derivative) = scalar_cost(index, x);
        sparse_gradient.set_at(0, sparse_gradient.at(0) + derivative);
        *sparse_hessian.coeff_ref(0, 0) += 1.0;
    }
    let gradient_at = |value| {
        (0..FACTOR_COUNT)
            .map(|index| scalar_cost(index, value).1)
            .sum::<f64>()
    };
    check_scalar_hessian(
        sparse_gradient.at(0),
        sparse_hessian.coeff(0, 0),
        x,
        cost,
        gradient_at,
    );

    let mut block_gradient = zeros_vector::<f64, tinyopt::Dynamic>(1);
    let mut block_hessian = BlockSparseMat::<ScalarBlock>::new();
    for index in 0..FACTOR_COUNT {
        let (_, derivative) = scalar_cost(index, x);
        block_gradient.set_at(0, block_gradient.at(0) + derivative);
        let mut block = block_hessian.block_mut::<1, 1>(0, 0);
        block.set_at(0, 0, block.at(0, 0) + 1.0);
    }
    check_scalar_hessian(
        block_gradient.at(0),
        block_hessian.get(0, 0),
        x,
        cost,
        gradient_at,
    );
}

#[test]
fn optimizes_sparse_and_block_sparse_factors() {
    let sparse_factor = |index,
                         x: &[f64],
                         gradient: Option<&mut Vector<f64, tinyopt::Dynamic>>,
                         hessian: Option<&mut SparseMatrix<f64>>| {
        let (cost, derivative) = scalar_cost(index, x[0]);
        if let Some(gradient) = gradient {
            gradient.set_at(0, gradient.at(0) + derivative);
        }
        if let Some(hessian) = hessian {
            *hessian.coeff_ref(0, 0) += 1.0;
        }
        Cost::with_num_residuals(cost, 1)
    };
    let block_factor = |index,
                        x: &[f64],
                        gradient: Option<&mut Vector<f64, tinyopt::Dynamic>>,
                        hessian: Option<&mut BlockSparseMat<ScalarBlock>>| {
        let (cost, derivative) = scalar_cost(index, x[0]);
        if let Some(gradient) = gradient {
            gradient.set_at(0, gradient.at(0) + derivative);
        }
        if let Some(hessian) = hessian {
            let mut block = hessian.block_mut::<1, 1>(0, 0);
            block.set_at(0, 0, block.at(0, 0) + 1.0);
        }
        Cost::with_num_residuals(cost, 1)
    };

    let mut sparse_x = vec![0.0_f64];
    let sparse_summary = optimize_parallel_sparse(
        &mut sparse_x,
        FACTOR_COUNT,
        sparse_factor,
        &Options::default(),
    );
    assert!(
        sparse_summary.converged(),
        "{:?}",
        sparse_summary.stop_reason
    );
    assert!((sparse_x[0] - 1.0).abs() < 1e-6);

    let mut block_x = vec![0.0_f64];
    let block_summary = optimize_parallel_block_sparse::<ScalarBlock, _, _>(
        &mut block_x,
        FACTOR_COUNT,
        block_factor,
        &Options::default(),
    );
    assert!(block_summary.converged(), "{:?}", block_summary.stop_reason);
    assert!((block_x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn optimizes_static_and_dynamic_parameters() {
    let fixed = optimize_problem([0.0, 0.0]);
    let dynamic = optimize_problem(vec![0.0, 0.0]);
    for params in [fixed.as_slice(), dynamic.as_slice()] {
        assert!((params[0] - 1.0).abs() < 1e-6);
        assert!((params[1] + 2.0).abs() < 1e-6);
    }
}

#[test]
#[should_panic(expected = "num_factors must be positive")]
fn rejects_empty_factor_sets() {
    let mut x = [0.0; 2];
    optimize_parallel(
        &mut x,
        0,
        |index, values, gradient, hessian| {
            quadratic_factor::<Static<2>>(index, values, gradient, hessian)
        },
        &Options::default(),
    );
}
