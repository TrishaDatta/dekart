// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

use crate::sigma_protocol::{homomorphism::fixed_base_msms, FirstProofItem};
use aptos_crypto::arkworks::{
    msm::MsmInput,
    random::{unsafe_random_point, UniformRand},
};
use aptos_crypto_derive::SigmaProtocolWitness;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{CryptoRng, RngCore};
use std::fmt::Debug;
use crate::{sigma_protocol, Scalar};

pub use crate::sigma_protocol::homomorphism::TrivialShape as CodomainShape;

pub type Proof<C> = sigma_protocol::Proof<
    <<C as CurveGroup>::Affine as AffineRepr>::ScalarField,
    Homomorphism<C>,
>;

impl<C: CurveGroup> Proof<C> {
    pub fn generate<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
        Self {
            first_proof_item: FirstProofItem::Commitment(CodomainShape(unsafe_random_point(
                rng,
            ))),
            z: Witness {
                poly_randomness: Scalar::rand(rng),
                hiding_kzg_randomness: Scalar::rand(rng),
            },
        }
    }
}

#[derive(CanonicalSerialize, Clone, Debug, PartialEq, Eq)]
pub struct Homomorphism<C: CurveGroup> {
    pub base_1: C::Affine,
    pub base_2: C::Affine,
}

#[derive(
    SigmaProtocolWitness, CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq,
)]
pub struct Witness<F: PrimeField> {
    pub poly_randomness: Scalar<F>,
    pub hiding_kzg_randomness: Scalar<F>,
}

impl<C: CurveGroup> crate::sigma_protocol::homomorphism::Trait for Homomorphism<C> {
    type Codomain = CodomainShape<C>;
    type CodomainNormalized = CodomainShape<C::Affine>;
    type Domain = Witness<C::ScalarField>;

    fn apply(&self, input: &Self::Domain) -> Self::Codomain {
        CodomainShape(
            self.base_1 * input.poly_randomness.0 + self.base_2 * input.hiding_kzg_randomness.0,
        )
    }

    fn normalize(&self, value: Self::Codomain) -> Self::CodomainNormalized {
        <Homomorphism<C> as fixed_base_msms::Trait>::normalize_output(value)
    }
}

impl<C: CurveGroup> fixed_base_msms::Trait for Homomorphism<C> {
    type Base = C::Affine;
    type CodomainShape<T>
        = CodomainShape<T>
    where
        T: CanonicalSerialize + CanonicalDeserialize + Clone + Eq + Debug;
    type MsmOutput = C;
    type Scalar = C::ScalarField;

    fn msm_terms(
        &self,
        input: &Self::Domain,
    ) -> Self::CodomainShape<MsmInput<Self::Base, Self::Scalar>> {
        let mut scalars = Vec::with_capacity(2);
        scalars.push(input.poly_randomness.0);
        scalars.push(input.hiding_kzg_randomness.0);

        let mut bases = Vec::with_capacity(2);
        bases.push(self.base_1);
        bases.push(self.base_2);

        CodomainShape(MsmInput { bases, scalars })
    }

    fn msm_eval(input: MsmInput<Self::Base, Self::Scalar>) -> Self::MsmOutput {
        C::msm(input.bases(), input.scalars()).expect("MSM failed in TwoTermMSM")
    }

    fn batch_normalize(msm_output: Vec<Self::MsmOutput>) -> Vec<Self::Base> {
        C::normalize_batch(&msm_output)
    }
}

impl<C: CurveGroup> sigma_protocol::CurveGroupTrait for Homomorphism<C> {
    type Group = C;

    fn dst(&self) -> Vec<u8> {
        b"DEKART_V2_SIGMA_PROTOCOL".to_vec()
    }
}
