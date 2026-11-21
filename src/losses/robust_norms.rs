// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Robust M-estimators, mirroring `tinyopt::losses::robust_norms`.
//!
//! Every loss is a function of the **squared** norm `n2` and of the **squared**
//! threshold `th2`. The `*_with_scale` variants additionally return the scale
//! to apply to the residual Jacobian, exactly like the C++ functions returning
//! `(loss, scale)` when passed `true`.

use crate::losses::norms::squared_l2;
use crate::math::sqrt;
use crate::real::Real;

/// Huber weight: `1` inside the threshold, `delta / ||x||` outside.
pub fn huber_weight<S: Real>(n2: S, delta: S) -> S {
    let delta2 = delta * delta;
    if n2 <= delta2 {
        S::one()
    } else {
        let floor = S::min_positive_value();
        delta / num_traits::Float::sqrt(if n2 > floor { n2 } else { floor })
    }
}

/// Huber loss value for a squared norm and a threshold `delta`.
pub fn huber_loss_value<S: Real>(n2: S, delta: S) -> S {
    let delta2 = delta * delta;
    if n2 <= delta2 {
        n2
    } else {
        let floor = S::min_positive_value();
        S::from_f64(2.0).expect("2 fits")
            * delta
            * num_traits::Float::sqrt(if n2 > floor { n2 } else { floor })
            - delta2
    }
}

/// Hard truncation: the loss is clipped at `th2`, the scale is `{0, 1}`.
pub fn truncated<S: Real>(n2: S, th2: S) -> S {
    if n2 <= th2 { n2 } else { th2 }
}

/// Hard truncation with its Jacobian scale.
pub fn truncated_with_scale<S: Real>(n2: S, th2: S) -> (S, S) {
    if n2 <= th2 {
        (n2, S::one())
    } else {
        (th2, S::zero())
    }
}

/// Huber loss on the squared norm.
pub fn huber<S: Real>(n2: S, th2: S) -> S {
    huber_with_scale(n2, th2).0
}

/// Huber loss and its Jacobian scale.
pub fn huber_with_scale<S: Real>(n2: S, th2: S) -> (S, S) {
    if n2 <= th2 {
        (n2, S::one())
    } else {
        let th = sqrt(th2);
        let n = sqrt(n2);
        let scale = num_traits::Float::max(S::min_positive_value(), th / n);
        (S::from_f64(2.0).expect("2 fits") * th * n - th2, scale)
    }
}

/// Tukey biweight loss (without the C++ `/6` factor, as in tinyopt).
pub fn tukey<S: Real>(n2: S, th2: S) -> S {
    tukey_with_scale(n2, th2).0
}

/// Tukey biweight loss and its Jacobian scale.
pub fn tukey_with_scale<S: Real>(n2: S, th2: S) -> (S, S) {
    if n2 <= th2 {
        let s = S::one() - n2 / th2;
        let l = th2 * (S::one() - s * s * s);
        let diff = th2 - n2;
        let scale = S::from_f64(3.0).expect("3 fits") * diff * diff / (th2 * th2);
        (l, scale)
    } else {
        (th2, S::zero())
    }
}

/// Arctan loss: `sqrt(th2) * atan(n2 / sqrt(th2))`.
pub fn arctan<S: Real>(n2: S, th2: S) -> S {
    arctan_with_scale(n2, th2).0
}

/// Arctan loss and its Jacobian scale.
pub fn arctan_with_scale<S: Real>(n2: S, th2: S) -> (S, S) {
    let th = sqrt(th2);
    let l = th * num_traits::Float::atan(n2 / th);
    let tmp = n2 * n2 / th2;
    let scale = num_traits::Float::max(S::min_positive_value(), S::one() / (tmp + S::one()));
    (l, scale)
}

/// Cauchy loss: `th2 * ln(1 + n2 / th2)`.
pub fn cauchy<S: Real>(n2: S, th2: S) -> S {
    cauchy_with_scale(n2, th2).0
}

/// Cauchy loss and its Jacobian scale.
pub fn cauchy_with_scale<S: Real>(n2: S, th2: S) -> (S, S) {
    let s = S::one() + n2 / th2;
    let l = th2 * num_traits::Float::ln(s);
    let scale = num_traits::Float::max(S::min_positive_value(), S::one() / s);
    (l, scale)
}

/// Geman-McClure loss: `n2 / (n2 + th2)`.
pub fn geman_mcclure<S: Real>(n2: S, th2: S) -> S {
    geman_mcclure_with_scale(n2, th2).0
}

/// Geman-McClure loss and its Jacobian scale.
pub fn geman_mcclure_with_scale<S: Real>(n2: S, th2: S) -> (S, S) {
    let e2 = n2 + th2;
    let l = n2 / e2;
    let scale = th2 / (e2 * e2);
    (l, scale)
}

/// Blake-Zisserman loss: `-ln(exp(-n2) + exp(-th2))`.
pub fn blake_zisserman<S: Real>(n2: S, th2: S) -> S {
    blake_zisserman_with_scale(n2, th2).0
}

/// Blake-Zisserman loss and its Jacobian scale.
pub fn blake_zisserman_with_scale<S: Real>(n2: S, th2: S) -> (S, S) {
    let epsilon = num_traits::Float::exp(-th2);
    let l = -num_traits::Float::ln(num_traits::Float::exp(-n2) + epsilon);
    let scale = S::one() / (epsilon * num_traits::Float::exp(n2) + S::one());
    (l, scale)
}

macro_rules! vector_loss {
    ($name:ident, $scalar:ident) => {
        /// Vector-residual variant, using the squared L2 norm of `x`.
        pub fn $name<S: Real>(x: &[S], th2: S) -> S {
            $scalar(squared_l2(x), th2)
        }
    };
    ($name:ident, $scalar:ident, jac) => {
        /// Vector-residual variant returning the loss and the Jacobian scale.
        pub fn $name<S: Real>(x: &[S], th2: S) -> (S, S) {
            $scalar(squared_l2(x), th2)
        }
    };
}

vector_loss!(truncated_loss, truncated);
vector_loss!(truncated_loss_with_scale, truncated_with_scale, jac);
vector_loss!(huber_loss, huber);
vector_loss!(huber_loss_with_scale, huber_with_scale, jac);
vector_loss!(tukey_loss, tukey);
vector_loss!(tukey_loss_with_scale, tukey_with_scale, jac);
vector_loss!(arctan_loss, arctan);
vector_loss!(arctan_loss_with_scale, arctan_with_scale, jac);
vector_loss!(cauchy_loss, cauchy);
vector_loss!(cauchy_loss_with_scale, cauchy_with_scale, jac);
vector_loss!(geman_mcclure_loss, geman_mcclure);
vector_loss!(geman_mcclure_loss_with_scale, geman_mcclure_with_scale, jac);
vector_loss!(blake_zisserman_loss, blake_zisserman);
vector_loss!(
    blake_zisserman_loss_with_scale,
    blake_zisserman_with_scale,
    jac
);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn huber_branches() {
        // Inlier.
        assert_eq!(huber(1.0_f64, 4.0), 1.0);
        assert_eq!(huber_with_scale(1.0, 4.0), (1.0, 1.0));
        // Outlier: n2 = 25, th2 = 4 -> 2*2*5 - 4 = 16, scale = 2/5.
        let (l, s) = huber_with_scale(25.0, 4.0);
        assert_relative_eq!(l, 16.0);
        assert_relative_eq!(s, 0.4);
    }

    #[test]
    fn huber_weight_and_value() {
        assert_eq!(huber_weight(1.0_f64, 2.0), 1.0);
        assert_relative_eq!(huber_weight(9.0, 2.0), 2.0 / 3.0);
        assert_eq!(huber_loss_value(1.0_f64, 2.0), 1.0);
        assert_relative_eq!(huber_loss_value(9.0, 2.0), 2.0 * 2.0 * 3.0 - 4.0);
    }

    #[test]
    fn tukey_branches() {
        let (l, s) = tukey_with_scale(1.0_f64, 4.0);
        // s = 1 - 1/4 = 0.75, l = 4*(1 - 0.75³) = 2.3125
        assert_relative_eq!(l, 2.3125);
        assert_relative_eq!(s, 3.0 * 9.0 / 16.0);
        assert_eq!(tukey_with_scale(9.0, 4.0), (4.0, 0.0));
    }

    #[test]
    fn other_losses_values() {
        assert_relative_eq!(truncated(9.0_f64, 4.0), 4.0);
        assert_eq!(truncated(3.0_f64, 4.0), 3.0);

        // Arctan: th = 2, l = 2 * atan(9/2).
        assert_relative_eq!(arctan(9.0_f64, 4.0), 2.0 * (4.5_f64).atan());

        // Cauchy: th2 * ln(1 + n2/th2) = 4 * ln(1 + 9/4).
        assert_relative_eq!(cauchy(9.0_f64, 4.0), 4.0 * (1.0 + 2.25_f64).ln());

        // Geman-McClure: 9 / 13.
        assert_relative_eq!(geman_mcclure(9.0_f64, 4.0), 9.0 / 13.0);

        // Blake-Zisserman: -ln(e^-9 + e^-4).
        let expected = -((-9.0_f64).exp() + (-4.0_f64).exp()).ln();
        assert_relative_eq!(blake_zisserman(9.0_f64, 4.0), expected);
    }

    #[test]
    fn vector_losses_use_squared_norm() {
        let x = [3.0_f64, 4.0];
        // ||x||² = 25, th2 = 4 -> same as the scalar outlier case.
        assert_relative_eq!(huber_loss(&x, 4.0), 16.0);
        assert_relative_eq!(cauchy_loss(&x, 4.0), cauchy(25.0, 4.0));
        assert_relative_eq!(tukey_loss(&x, 4.0), tukey(25.0, 4.0));
    }
}
