// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License.
// Keccak-based Fiat–Shamir transcript for sumcheck (from Jolt, simplified).

use crate::sumcheck::field::SumcheckField;
use ark_serialize::CanonicalSerialize;
use sha3::{Digest, Keccak256};

/// Transcript interface for sumcheck so that callers (e.g. range proofs) can drive
/// the sumcheck with their own transcript (e.g. Merlin).
pub trait SumcheckTranscript {
    fn append_scalar<F: SumcheckField + CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        scalar: &F,
    );
    fn append_scalars<F: SumcheckField + CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        scalars: &[F],
    );
    fn challenge_scalar<F: SumcheckField + ark_ff::PrimeField>(&mut self) -> F;
    fn challenge_vector<F: SumcheckField + ark_ff::PrimeField>(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.challenge_scalar::<F>()).collect()
    }
}

#[derive(Clone, Default)]
pub struct KeccakSumcheckTranscript {
    state: [u8; 32],
    n_rounds: u32,
}

impl KeccakSumcheckTranscript {
    pub fn new(label: &'static [u8]) -> Self {
        let mut hasher = Keccak256::new();
        hasher.update(label);
        let state: [u8; 32] = hasher.finalize().into();
        Self { state, n_rounds: 0 }
    }

    fn hasher(&self) -> Keccak256 {
        let mut packed = [0u8; 36];
        packed[0..32].copy_from_slice(&self.state);
        packed[32..36].copy_from_slice(&self.n_rounds.to_be_bytes());
        let mut h = Keccak256::new();
        h.update(packed);
        h
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds = self.n_rounds.wrapping_add(1);
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining = out.len();
        let mut start = 0usize;
        while remaining > 32 {
            let mut buf = [0u8; 32];
            let h = self.hasher().finalize();
            buf.copy_from_slice(&h);
            self.update_state(buf);
            out[start..start + 32].copy_from_slice(&buf);
            start += 32;
            remaining -= 32;
        }
        let h = self.hasher().finalize();
        self.update_state(h.into());
        out[start..].copy_from_slice(&self.state[..remaining]);
    }

    /// Append a single scalar to the transcript (internal; use SumcheckTranscript trait).
    fn do_append_scalar<F: SumcheckField + CanonicalSerialize>(&mut self, scalar: &F) {
        let mut buf = vec![0u8; 32];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        let mut hasher = self.hasher();
        hasher.update(&buf);
        self.update_state(hasher.finalize().into());
    }

    /// Append multiple scalars (internal).
    fn do_append_scalars<F: SumcheckField + CanonicalSerialize>(&mut self, scalars: &[F]) {
        let mut buf = vec![];
        for s in scalars {
            s.serialize_uncompressed(&mut buf).unwrap();
        }
        let mut hasher = self.hasher();
        hasher.update(&buf);
        self.update_state(hasher.finalize().into());
    }

    /// Produce one field element as challenge (internal).
    fn do_challenge_scalar<F: SumcheckField + ark_ff::PrimeField>(&mut self) -> F {
        let mut buf = [0u8; 32];
        self.challenge_bytes(&mut buf);
        F::from_be_bytes_mod_order(&buf)
    }

    /// Produce a vector of field elements (internal).
    fn do_challenge_vector<F: SumcheckField + ark_ff::PrimeField>(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.do_challenge_scalar::<F>()).collect()
    }
}

impl SumcheckTranscript for KeccakSumcheckTranscript {
    fn append_scalar<F: SumcheckField + CanonicalSerialize>(
        &mut self,
        _label: &'static [u8],
        scalar: &F,
    ) {
        self.do_append_scalar(scalar);
    }

    fn append_scalars<F: SumcheckField + CanonicalSerialize>(
        &mut self,
        _label: &'static [u8],
        scalars: &[F],
    ) {
        self.do_append_scalars(scalars);
    }

    fn challenge_scalar<F: SumcheckField + ark_ff::PrimeField>(&mut self) -> F {
        self.do_challenge_scalar::<F>()
    }

    fn challenge_vector<F: SumcheckField + ark_ff::PrimeField>(&mut self, len: usize) -> Vec<F> {
        self.do_challenge_vector::<F>(len)
    }
}
