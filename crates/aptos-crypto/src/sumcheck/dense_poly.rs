// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License.
// Dense multilinear polynomial with binding (from Jolt, no jolt dep).

use crate::sumcheck::field::SumcheckField;

#[derive(Clone, Debug)]
pub struct DensePolynomial<F: SumcheckField> {
    pub num_vars: usize,
    pub len: usize,
    pub Z: Vec<F>,
}

#[derive(Clone, Copy, Debug)]
pub enum BindingOrder {
    LowToHigh,
    HighToLow,
}

impl<F: SumcheckField> DensePolynomial<F> {
    pub fn new(Z: Vec<F>) -> Self {
        assert!(
            Z.len().is_power_of_two(),
            "Dense multilinear polynomial must have power-of-2 length"
        );
        let num_vars = Z.len().trailing_zeros() as usize;
        let len = Z.len();
        Self { num_vars, len, Z }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Bind the top variable: fold Z into half length (left + r * (right - left)).
    pub fn bind(&mut self, r: &F::Challenge, _order: BindingOrder) {
        let r_f = F::challenge_to_field(r);
        let n = self.len / 2;
        let (left, right) = self.Z.split_at_mut(n);
        for (a, b) in left.iter_mut().zip(right.iter()) {
            *a += r_f * (*b - *a);
        }
        self.num_vars -= 1;
        self.len = n;
    }
}
