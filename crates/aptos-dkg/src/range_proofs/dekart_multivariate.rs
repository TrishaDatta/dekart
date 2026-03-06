// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

use super::scalars_to_bits;
use crate::{
    fiat_shamir::{PolynomialCommitmentScheme, RangeProof},
    pcs::{
        shplonked::{
            batch_open_generalized, batch_pairing_for_verify_generalized, Shplonked,
            ShplonkedBatchProof, Srs, SumEvalHom,
        },
        traits::PolynomialCommitmentScheme as _,
        univariate_hiding_kzg,
        zeromorph::{replay_challenges, zeta_z_com, Zeromorph, ZeromorphProverKey},
        EvaluationSet,
    },
    pvss::chunky::chunked_elgamal::correlated_randomness,
    range_proofs::{dekart_univariate_v2::two_term_msm, traits, PublicStatement},
    sigma_protocol::{
        homomorphism::{Trait as _, TrivialShape},
        Trait as _,
    },
    sumcheck::merlin_transcript::MerlinSumcheckTranscript,
    Scalar,
};
use aptos_crypto::{
    arkworks::{
        msm::{msm_bool, MsmInput},
        random::{sample_field_element, sample_field_elements},
        srs::{SrsBasis, SrsType},
        GroupGenerators,
    },
    sumcheck::{
        BatchedSumcheck, BooleanityEqSumcheckProverLSB,
        BooleanityEqSumcheckVerifierLSBWithOpenings, ClearSumcheckProof, MaskingPolynomial,
        ProverOpeningAccumulator, UniPoly, VerifierOpeningAccumulator,
    },
    utils::powers,
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{AdditiveGroup, Field};
use ark_poly::{
    univariate::DensePolynomial, DenseMultilinearExtension, DenseUVPolynomial, Polynomial,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{CryptoRng, RngCore};
#[cfg(feature = "range_proof_timing_multivariate")]
use std::time::{Duration, Instant};
use std::{fmt::Debug, iter::once};

#[allow(non_snake_case)]
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct ProverKey<E: Pairing> {
    pub(crate) vk: VerificationKey<E>,
    pub(crate) ck: univariate_hiding_kzg::CommitmentKey<E>,
    pub(crate) max_n: usize,
    //pub(crate) prover_precomputed: ProverPrecomputed<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct VerificationKey<E: Pairing> {
    xi_1: E::G1Affine,
    last_tau: E::G1Affine,
    vk_hkzg: univariate_hiding_kzg::VerificationKey<E>,
    /// Number of variables for sumcheck (log2 of domain size)
    pub(crate) num_variables: usize,
    srs: Srs<E>,
}

#[allow(non_snake_case)]
#[derive(CanonicalSerialize, Clone, CanonicalDeserialize)]
pub struct Proof<E: Pairing> {
    /// Blinding commitment C_β (None if blinding was not used)
    pub blinding_poly_comm: Option<E::G1Affine>,
    /// Proof that C_β is of the form β·eq_0 (None if blinding was not used)
    pub blinding_poly_proof: Option<two_term_msm::Proof<E::G1>>,
    pub f_j_commitments: Vec<E::G1Affine>,
    pub g_i_commitments: Vec<E::G1Affine>,
    pub H_g: E::ScalarField,
    pub sumcheck_proof: ClearSumcheckProof<E::ScalarField>,
    pub y_f: E::ScalarField,
    pub y_js: Vec<E::ScalarField>,
    pub y_g: E::ScalarField,
    pub zk_pcs_batch_proof: ShplonkedBatchProof<E>,
    pub shplonked_eval_points: Vec<E::ScalarField>,
    pub zeromorph_q_hat_com: E::G1Affine,
    pub zeromorph_q_k_com: Vec<E::G1Affine>,
}

#[allow(non_snake_case)]
impl<E: Pairing> traits::BatchedRangeProof<E> for Proof<E> {
    type Commitment = univariate_hiding_kzg::Commitment<E>;
    type CommitmentKey = univariate_hiding_kzg::CommitmentKey<E>;
    type CommitmentNormalised = univariate_hiding_kzg::CommitmentNormalised<E>;
    type CommitmentRandomness = univariate_hiding_kzg::CommitmentRandomness<E::ScalarField>;
    type Input = E::ScalarField;
    type ProverKey = ProverKey<E>;
    type PublicStatement = PublicStatement<E>;
    type VerificationKey = VerificationKey<E>;

    /// Domain-separation tag (DST) used to ensure that all cryptographic hashes and
    /// transcript operations within the protocol are uniquely namespaced
    const DST: &[u8] = b"MULTIVARIATE_DEKART_RANGE_PROOF_DST";

    fn maul(&mut self) {
        if let Some(c) = self.f_j_commitments.first_mut() {
            *c = (c.into_group() + E::G1::generator()).into_affine();
        }
    }

    fn commitment_key_from_prover_key(pk: &Self::ProverKey) -> Self::CommitmentKey {
        pk.ck.clone()
    }

    fn setup<R: RngCore + CryptoRng>(
        max_n: usize,
        _max_ell: u8,
        group_generators: GroupGenerators<E>,
        rng: &mut R,
    ) -> (ProverKey<E>, VerificationKey<E>) {
        let size = (max_n + 1).next_power_of_two();
        let trapdoor = univariate_hiding_kzg::Trapdoor::<E>::rand(rng);
        let (vk_hkzg, ck) = univariate_hiding_kzg::setup(
            size,
            SrsType::PowersOfTau,
            group_generators.clone(),
            trapdoor,
        );
        let tau_powers = match &ck.msm_basis {
            SrsBasis::PowersOfTau { tau_powers } => tau_powers.clone(),
            _ => panic!("Expected PowersOfTau SRS"),
        };
        // VK only stores the first `size` tau powers (minimal needed for num_vars); prover uses ck for full SRS.
        let vk_taus_1: Vec<E::G1Affine> = tau_powers[..size].to_vec();
        let last_tau = *vk_taus_1
            .last()
            .expect("PowersOfTau SRS has at least one element");
        let num_variables = size.ilog2() as usize;
        let srs = Srs {
            taus_1: vk_taus_1,
            xi_1: ck.xi_1,
            g_2: vk_hkzg.group_generators.g2,
            tau_2: vk_hkzg.tau_2,
            xi_2: vk_hkzg.xi_2,
        };
        let vk = VerificationKey {
            xi_1: ck.xi_1,
            last_tau,
            vk_hkzg,
            num_variables,
            srs,
        };
        let pk = ProverKey {
            vk: vk.clone(),
            ck,
            max_n,
        };
        (pk, vk)
    }

    fn commit_with_randomness(
        ck: &Self::CommitmentKey,
        values: &[Self::Input],
        rho: &Self::CommitmentRandomness,
    ) -> Self::Commitment {
        // Multilinear extension of (β, z_1,…,z_n); β=0 when blinding deferred to ZKSC.Blind.
        // Layout: coeffs[0]=β, coeffs[1..=n]=values, rest zero. Size = smallest 2^m with 2^m ≥ n+1.
        let size = (values.len() + 1).next_power_of_two();
        let mut coeffs = Vec::with_capacity(size);
        coeffs.push(E::ScalarField::ZERO); // so it's always zero
        coeffs.extend_from_slice(values);
        coeffs.resize(size, E::ScalarField::ZERO);
        univariate_hiding_kzg::commit_with_randomness(ck, &coeffs, rho)
    }

    fn pairing_for_verify<R: RngCore + CryptoRng>(
        &self,
        vk: &Self::VerificationKey,
        n: usize,
        ell: u8,
        comm: &Self::Commitment,
        rng: &mut R,
    ) -> anyhow::Result<(Vec<E::G1Affine>, Vec<E::G2Affine>)> {
        #[cfg(feature = "range_proof_timing_multivariate")]
        let mut cumulative = Duration::ZERO;
        #[cfg(feature = "range_proof_timing_multivariate")]
        let mut print_cumulative = |name: &str, duration: Duration| {
            cumulative += duration;
            println!(
                "{:>10.2} ms  ({:>10.2} ms cum.)  {}",
                duration.as_secs_f64() * 1000.0,
                cumulative.as_secs_f64() * 1000.0,
                name
            );
        };

        // Number of variables for this instance (must match prover; can be less than vk max)
        let num_vars = (n + 1).next_power_of_two().ilog2() as usize;
        if num_vars > vk.num_variables {
            anyhow::bail!(
                "instance n={} requires num_vars={} but setup supports at most {}",
                n,
                num_vars,
                vk.num_variables
            );
        }

        // Step 1c: Append initial data to the Fiat-Shamir transcript.
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let mut trs = merlin::Transcript::new(b"dekart_multivariate");
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_vk(&mut trs, &vk);
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_public_statement(
            &mut trs,
            PublicStatement {
                n, // TODO: do we want to append the actual n or the max_n? Or its log?
                ell,
                comm: comm.clone(),
            },
        );
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("transcript init (vk + public statement)", start.elapsed());

        // Step 2: Verify the blinding commitment
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        if let (Some(blinding_comm), Some(blinding_proof)) =
            (&self.blinding_poly_comm, &self.blinding_poly_proof)
        {
            let hom = two_term_msm::Homomorphism {
                base_1: vk.last_tau,
                base_2: vk.xi_1,
            };
            hom.verify(
                &TrivialShape((*blinding_comm).into()),
                blinding_proof,
                &(),
                Some(1),
                rng,
            )?;
        }
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("blinding two_term_msm verify", start.elapsed());

        // Step 3a–3d: Append commitments and draw challenges
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_f_j_commitments(
            &mut trs,
            &self.f_j_commitments,
        );
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_g_i_commitments(
            &mut trs,
            &self.g_i_commitments,
        );
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_hypercube_sum(&mut trs, &self.H_g);

        let c: E::ScalarField =
            <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_scalar(&mut trs);
        let alpha: E::ScalarField =
            <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_nonzero_scalar(&mut trs);
        let c_zc = <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_point(
            &mut trs,
            num_vars as u8,
        );
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative(
            "append commitments + challenges (c, alpha, c_zc)",
            start.elapsed(),
        );

        // Step 3e: Sumcheck verify (aptos-crypto BooleanityEq LSB with alpha*g masking)
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let claimed_sum = alpha * self.H_g;
        let alpha_y_g = alpha * self.y_g;
        let verifier = BooleanityEqSumcheckVerifierLSBWithOpenings::new_with_alpha_y_g(
            num_vars,
            c,
            c_zc.clone(),
            claimed_sum,
            self.y_js[..ell as usize].to_vec(),
            Some(alpha_y_g),
        );
        let mut sumcheck_transcript = MerlinSumcheckTranscript::new(&mut trs);
        let mut verifier_acc = VerifierOpeningAccumulator::new(0, false);
        let mut sumcheck_diagnostics = Vec::new();
        let r_sumcheck = BatchedSumcheck::verify_standard_with_diagnostics::<E::ScalarField, _>(
            &self.sumcheck_proof,
            vec![&verifier],
            &mut verifier_acc,
            &mut sumcheck_transcript,
            &mut sumcheck_diagnostics,
        );
        let r_sumcheck = r_sumcheck.map_err(|e| {
            eprintln!("sumcheck verify failed: {:?}. Diagnostics:", e);
            for line in &sumcheck_diagnostics {
                eprintln!("  {}", line);
            }
            anyhow::anyhow!("sumcheck verify: {:?}", e)
        })?;
        let x: Vec<E::ScalarField> = r_sumcheck;
        let subclaim_expected_eval = {
            let eq_c_zc_at_x: E::ScalarField = (0..x.len())
                .map(|i| {
                    let c_i = c_zc[i];
                    let xi = x[i];
                    (E::ScalarField::ONE - c_i) + xi * (c_i + c_i - E::ScalarField::ONE)
                })
                .product();
            let eq_zero_at_x: E::ScalarField =
                x.iter().map(|&xi| E::ScalarField::ONE - xi).product();
            let z_0_at_x = E::ScalarField::ONE - eq_zero_at_x;
            let mut sum_c_j = E::ScalarField::ZERO;
            let mut c_pow = c;
            for &y_j in self.y_js.iter().take(ell as usize) {
                sum_c_j += c_pow * y_j * (E::ScalarField::ONE - y_j);
                c_pow *= c;
            }
            sum_c_j * eq_c_zc_at_x * z_0_at_x + alpha * self.y_g
        };

        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("sumcheck verify", start.elapsed());

        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();

        // Step 4a: (sum c^j y_j(1-y_j)) * eq_c_zc(x) * Z_0(x) + alpha * y_g == h_m(x_m)
        // BinaryConstraintPolynomial uses (1 - eq_0) with eq_0 = ∏ᵢ(1-Xᵢ); Z_0 = 1 - eq_0 (vanishes at (0,...,0)).
        // Use the same variable order as the sumcheck prover: subclaim.point is (x_0,…,x_{m-1}) with x_i
        // from round i+1; eq_point c_zc is indexed to match. So we use x as-is for eq and Z_0. // OLD comment: // DenseMultilinearExtension in ark_poly uses index = sum_i b_i * 2^i with b_0 LSB, so var 0 = LSB. Match that.
        let eq_c_zc_at_x: E::ScalarField = (0..x.len())
            .map(|i| {
                let c_i = c_zc[i];
                let xi = x[i];
                (E::ScalarField::ONE - c_i) + xi * (c_i + c_i - E::ScalarField::ONE)
            })
            .product();
        let eq_zero_at_x: E::ScalarField = x.iter().map(|&xi| E::ScalarField::ONE - xi).product();
        let Z_0_at_x = E::ScalarField::ONE - eq_zero_at_x;

        let mut sum_c_j_y_j_one_minus_y_j = E::ScalarField::ZERO;
        let mut c_pow = c;
        for &y_j in self.y_js.iter().take(ell as usize) {
            sum_c_j_y_j_one_minus_y_j += c_pow * y_j * (E::ScalarField::ONE - y_j);
            c_pow *= c;
        }
        let lhs_4a = sum_c_j_y_j_one_minus_y_j * eq_c_zc_at_x * Z_0_at_x + alpha * self.y_g;
        if lhs_4a != subclaim_expected_eval {
            return Err(anyhow::anyhow!(
                "Step 4a check failed: (sum c^j y_j(1-y_j)) * eq_c_zc(x) * Z_0(x) + alpha*y_g != h(x). \
                 Check transcript order (vk, C, n, ell, C_fj, C_gi, H_g, c, alpha, c_zc), \
                 that y_js/y_g are f_j(x)/g(x) at subclaim.point, and sumcheck variable order."
            ));
        }

        // Step 4b: y_f - sum_{j=1}^ℓ 2^{j-1} y_j == 0 (correlated masks)
        let two = E::ScalarField::from(2u64);
        let mut pow2 = E::ScalarField::ONE;
        let mut sum_2j_minus_1_y_j = E::ScalarField::ZERO;
        for &y_j in self.y_js.iter().take(ell as usize) {
            sum_2j_minus_1_y_j += pow2 * y_j;
            pow2 *= two;
        }
        if self.y_f != sum_2j_minus_1_y_j {
            return Err(anyhow::anyhow!(
                "Step 4b check failed: y_f - sum_{{j=1}}^ℓ 2^{{j-1}} y_j != 0 (correlated masks)."
            ));
        }
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("step4 scalar check (eq_c_zc, Z_0, lhs)", start.elapsed());

        // Step 5a: Add y_f and {y_j}_{1≤j≤ℓ} to the Fiat–Shamir transcript.        #[cfg(feature = "range_proof_timing_multivariate")]
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        trs.append_evaluation_points(&[self.y_f]);
        for &y_j in self.y_js.iter().take(ell as usize) {
            trs.append_evaluation_points(&[y_j]);
        }

        // Step 5b: Challenge hat_c.
        let hat_c: E::ScalarField =
            <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_scalar(&mut trs);
        let hat_c_powers = powers(hat_c, ell as usize + 1);

        // Step 5c: Replay mPCS.ReduceToUnivariate (Zeromorph) to get x_challenge and form zeta_z_com.
        let (y_challenge, x_challenge, z_challenge) =
            replay_challenges::<E>(&mut trs, &self.zeromorph_q_k_com, &self.zeromorph_q_hat_com);

        anyhow::ensure!(
            self.shplonked_eval_points[0] == x_challenge,
            "Batched opening: first eval point must equal Zeromorph x_challenge"
        );

        let batched_eval = self.y_f
            + self.y_js[0..ell as usize]
                .iter()
                .enumerate()
                .map(|(j, &y_j)| hat_c_powers[j + 1] * y_j)
                .sum::<E::ScalarField>();

        // Now form the MSM corresponding to batching the Zeromorph openings
        let mut combined_bases = vec![comm.0.into_affine()];
        let mut combined_scalars = vec![E::ScalarField::ONE];
        if let Some(ref bc) = self.blinding_poly_comm {
            combined_bases.push(*bc);
            combined_scalars.push(E::ScalarField::ONE);
        }
        combined_bases.extend(self.f_j_commitments.iter().copied());
        combined_scalars.extend(hat_c_powers.iter().skip(1).copied());
        let combined_comm =
            E::G1::msm(&combined_bases, &combined_scalars).expect("combined commitment MSM");

        let point_reversed: Vec<E::ScalarField> = x.iter().rev().cloned().collect();
        let zeromorph_msm = zeta_z_com::<E>(
            self.zeromorph_q_hat_com,
            combined_comm.into_affine(),
            vk.vk_hkzg.group_generators.g1,
            &self.zeromorph_q_k_com,
            y_challenge,
            x_challenge,
            z_challenge,
            &point_reversed,
            batched_eval,
        );

        let g_commitment_msms: Vec<MsmInput<E::G1Affine, E::ScalarField>> = self
            .g_i_commitments
            .iter()
            .map(|&affine| {
                MsmInput::new(vec![affine], vec![E::ScalarField::ONE]).expect("single term")
            })
            .collect();
        let commitment_msms: Vec<MsmInput<E::G1Affine, E::ScalarField>> =
            once(zeromorph_msm).chain(g_commitment_msms).collect();
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative(
            "transcript y_f,y_js + hat_c + replay_challenges + combined_comm + zeta_z_com",
            start.elapsed(),
        );

        // Step 5d: Build sets, y_rev, then single uPCS.BatchVerify
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let sets: Vec<EvaluationSet<E::ScalarField>> = self
            .shplonked_eval_points
            .iter()
            .enumerate()
            .map(|(i, &z)| {
                if i == 0 {
                    EvaluationSet {
                        rev: vec![z],
                        hid: vec![],
                    }
                } else {
                    EvaluationSet {
                        rev: vec![],
                        hid: vec![z],
                    }
                }
            })
            .collect();

        // First poly (Zeromorph) vanishes at first point; rest have hidden evals only.
        let y_rev: Vec<Vec<E::ScalarField>> = (0..sets.len())
            .map(|i| {
                if i == 0 {
                    vec![E::ScalarField::ZERO]
                } else {
                    vec![]
                }
            })
            .collect();
        let (g1_terms, g2_terms) =
            batch_pairing_for_verify_generalized::<E, _, SumEvalHom<E::ScalarField>>(
                &vk.srs,
                &sets,
                &SumEvalHom::<E::ScalarField>::default(),
                &commitment_msms,
                &y_rev,
                self.zk_pcs_batch_proof.sigma_proof_statement.1,
                &self.zk_pcs_batch_proof,
                &mut trs,
                rng,
            )?;
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("batch_pairing_for_verify_generalized", start.elapsed());

        Ok((g1_terms, g2_terms))
    }

    #[allow(non_snake_case)]
    fn prove<R: RngCore + CryptoRng>(
        pk: &ProverKey<E>,
        values: &[Self::Input],
        ell: u8,
        comm: &Self::CommitmentNormalised,
        rho: &Self::CommitmentRandomness,
        rng: &mut R,
    ) -> Proof<E> {
        // Use blinding=false: with blinding, combined_comm = comm + comm_blinding_poly + ... mixes
        // a two-term MSM (beta*tau^{n-1}+rho*xi) with KZG commitments; that sum is not a KZG
        // commitment to batched_coeffs, so the batched opening verification fails.
        let comm_conv = TrivialShape(comm.0.into_group()); // TODO: hacky, remove etc
        prove_impl(pk, values, ell, &comm_conv, rho, rng, false)
    }
}

/// Prover with optional blinding. When `use_blinding` is false, β=0 and no C_β is produced.
#[allow(non_snake_case)]
pub fn prove_impl<E: Pairing, R: RngCore + CryptoRng>(
    pk: &ProverKey<E>,
    values: &[E::ScalarField],
    ell: u8,
    comm: &univariate_hiding_kzg::Commitment<E>,
    rho: &univariate_hiding_kzg::CommitmentRandomness<E::ScalarField>,
    rng: &mut R,
    use_blinding: bool,
) -> Proof<E> {
    #[cfg(feature = "range_proof_timing_multivariate")]
    let mut cumulative = Duration::ZERO;
    #[cfg(feature = "range_proof_timing_multivariate")]
    let mut print_cumulative = |name: &str, duration: Duration| {
        cumulative += duration;
        println!(
            "  {:>10.2} ms  ({:>10.2} ms cum.)  [dekart_multivariate prove] {}",
            duration.as_secs_f64() * 1000.0,
            cumulative.as_secs_f64() * 1000.0,
            name
        );
    };

    let mut trs = merlin::Transcript::new(b"dekart_multivariate");
    let tau_powers = match &pk.ck.msm_basis {
        SrsBasis::PowersOfTau { tau_powers } => tau_powers,
        _ => panic!("Expected PowersOfTau SRS"),
    };

    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_vk(&mut trs, &pk.vk);
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_public_statement(
        &mut trs,
        PublicStatement {
            n: values.len(),
            ell,
            comm: comm.clone(),
        },
    );

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 2 (optional)
    let (beta, comm_blinding_poly, comm_blinding_poly_rand, beta_sigma_proof) = if use_blinding {
        let last_msm_elt = tau_powers.last().expect("PowersOfTau SRS has no elements");
        let (b, c, r, proof): (
            E::ScalarField,
            E::G1,
            E::ScalarField,
            two_term_msm::Proof<E::G1>,
        ) = zksc_blind::<E, _>(*last_msm_elt, pk.ck.xi_1, rng);
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_blinding_poly_commitment(
            &mut trs, &c,
        );
        (b, Some(c), Some(r), Some(proof))
    } else {
        (
            E::ScalarField::ZERO,
            None,
            None,
            None::<two_term_msm::Proof<E::G1>>,
        )
    };
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("blinding (zksc_blind + transcript)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 3a: construct the bits for the f_js
    let bits = scalars_to_bits::scalars_to_bits_le(values, ell);
    let f_j_evals_without_r = scalars_to_bits::transpose_bit_matrix(&bits);

    // Step 3b: Sample correlated masks β_1,…,β_ℓ with β = Σ_{j=1}^ℓ 2^{j-1} β_j
    let betas: Vec<E::ScalarField> = correlated_randomness(rng, 2, u32::from(ell), &beta);

    // Step 3c: Construct f_j
    let size = (values.len() + 1).next_power_of_two();
    let num_vars = size.ilog2() as u8;
    let f_j_evals: Vec<Vec<E::ScalarField>> = f_j_evals_without_r
        .iter()
        .enumerate()
        .map(|(j, col)| {
            let mut evals: Vec<E::ScalarField> = once(betas[j])
                .chain(col.iter().map(|&b| E::ScalarField::from(b)))
                .collect();
            evals.resize(size, E::ScalarField::ZERO); // This is needed for `DenseMultilinearExtension::from_evaluations_vec`
            evals
        })
        .collect();

    let f_js: Vec<DenseMultilinearExtension<E::ScalarField>> = f_j_evals
        .iter()
        .map(|f_j_eval| {
            DenseMultilinearExtension::from_evaluations_vec(num_vars.into(), f_j_eval.clone())
        })
        .collect();
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("betas + bits + hat_f_j_evals + hat_f_js", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 3d:
    let f_j_comms_randomness: Vec<E::ScalarField> = sample_field_elements(f_js.len(), rng);
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("hat_f_j_comms_randomness (sample)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 3e: Commit to the f_js.
    // Homomorphism does: xi_1*r + sum_{i=0..size-1} tau_powers[i]*values[i]
    let f_j_comms_proj: Vec<E::G1> = f_j_evals
        .iter()
        .zip(f_j_comms_randomness.iter())
        .enumerate()
        .map(|(j, (f_j_eval, r_i))| {
            let bits = f_j_evals_without_r[j].clone();
            let sum = tau_powers[0] * f_j_eval[0] + msm_bool(&tau_powers[1..=values.len()], &bits);
            pk.ck.xi_1 * *r_i + sum // TODO: could turn this into a 3-term MSM, should be faster
        })
        .collect();
    let f_j_comms = E::G1::normalize_batch(&f_j_comms_proj);
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("hat_f_j commitments (hom.apply loop)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 3f:
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_f_j_commitments(&mut trs, &f_j_comms);

    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("transcript append hat_f_j_comms", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 4a:
    let srs = Srs {
        taus_1: tau_powers.clone(),
        xi_1: pk.ck.xi_1,
        g_2: pk.vk.vk_hkzg.group_generators.g2,
        tau_2: pk.vk.vk_hkzg.tau_2,
        xi_2: pk.vk.vk_hkzg.xi_2,
    };
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("SRS build", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let (g_is, g_i_commitments_proj, g_comm_randomnesses, H_g): (
        Vec<Vec<E::ScalarField>>,
        Vec<E::G1>,
        Vec<E::ScalarField>,
        E::ScalarField,
    ) = zksc_send_mask(&srs, 4, num_vars, rng);
    let g_j_comms = E::G1::normalize_batch(&g_i_commitments_proj);

    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("zksc_send_mask (g_is, g_comm, G)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 4b: Add {C_{g_i}} and H_g to the Fiat–Shamir transcript. // TODO: maybe combine with 3f for better batching
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_g_i_commitments(&mut trs, &g_j_comms);
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_hypercube_sum(&mut trs, &H_g);
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("transcript append g_comm + H_g", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 5a–5c: Verifier challenges c, alpha; eq_point t; run sumcheck on transcript with linear term (f - sum 2^{j-1} f_j) + sum c^j f_j(f_j-1)
    let c: E::ScalarField =
        <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_scalar(&mut trs);
    let alpha: E::ScalarField =
        <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_nonzero_scalar(&mut trs);
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("transcript challenges (c, alpha)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let mut f_evals = vec![E::ScalarField::ZERO; size];
    f_evals[0] = beta;
    for (i, &v) in values.iter().enumerate() {
        f_evals[i + 1] = v;
    }
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("f_evals construction", start.elapsed());

    // TODO: define hat(f) hier ipv in zkzc_send_polys()
    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let sumcheck_proof = zkzc_send_polys::<E>(
        &mut trs,
        g_is.clone(),
        num_vars,
        ell as usize,
        c,
        alpha,
        &f_j_evals,
        #[cfg(feature = "range_proof_timing_multivariate")]
        Some(&mut print_cumulative),
        #[cfg(not(feature = "range_proof_timing_multivariate"))]
        None,
    );
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("zkzc_send_polys (sumcheck total)", start.elapsed());

    // Sumcheck point: round challenges from aptos-crypto BatchedSumcheck::prove.
    let xs: Vec<E::ScalarField> = sumcheck_proof.1;
    debug_assert_eq!(xs.len(), num_vars as usize);

    // Step 6: Evaluations y_f = f(x), y_j = f_j(x) at sumcheck point x = (x_1,...,x_n)

    // Step 6a:
    let f_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars.into(), f_evals.clone());
    let y_f = f_poly.evaluate(&xs);

    // Step 6b:
    debug_assert_eq!(f_js.len(), ell as usize);
    let y_js: Vec<E::ScalarField> = f_js.iter().map(|f_j| f_j.evaluate(&xs)).collect();

    // Step 6c (spec): Add y_f and {y_j}_{1≤j≤ℓ} to the Fiat–Shamir transcript.
    trs.append_evaluation_points(&[y_f]);
    for &y_j in y_js.iter().take(ell as usize) {
        trs.append_evaluation_points(&[y_j]);
    }
    let hat_c: E::ScalarField =
        <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_scalar(&mut trs);
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative(
        "asserted_sum + xs + g_evals + y_g + sumcheck_point + y_f + y_evals + hat_c",
        start.elapsed(),
    );

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 6e:
    // Batched polynomial f̂ = f + sum_j hat_c^j f_j (coefficient form for univariate opening)
    let hat_c_powers = powers(hat_c, ell as usize + 1);
    let mut batched_evals = f_evals.clone();
    for j in 0..ell as usize {
        let cj = hat_c_powers[j + 1];
        for (i, b) in batched_evals.iter_mut().enumerate() {
            *b += cj * f_j_evals[j][i];
        }
    }

    // let z: E::ScalarField = trs.challenge_scalar();
    // let y: E::ScalarField = batched_evals
    //     .iter()
    //     .enumerate()
    //     .fold(E::ScalarField::ZERO, |acc, (i, &coeff)| {
    //         acc + coeff * z.pow([i as u64])
    //     });

    // TODO: not used??????????
    // Verifier will recompute this for the single batched zk_pcs_verify
    // let _combined_comm = {
    //     let mut bases = vec![comm.0.into_affine()];
    //     let mut scalars = vec![E::ScalarField::ONE];
    //     if let Some(ref cp) = comm_blinding_poly {
    //         bases.push((*cp).into_affine());
    //         scalars.push(E::ScalarField::ONE);
    //     }
    //     for (j, &cf) in hat_f_j_comms.iter().enumerate() {
    //         bases.push(cf.into_affine());
    //         scalars.push(hat_c_powers[j + 1]);
    //     }
    //     E::G1::msm(&bases, &scalars).expect("batched commitment MSM")
    // };

    let mut batched_randomness = rho.0 + comm_blinding_poly_rand.unwrap_or(E::ScalarField::ZERO);
    for (j, &r_j) in f_j_comms_randomness.iter().enumerate() {
        batched_randomness += hat_c_powers[j + 1] * r_j;
    }

    // Step 6g (spec): mPCS.ReduceToUnivariate(x, f̂, ρ̂) = Zeromorph::open_to_batched_instance.
    // Produces batched univariate poly that vanishes at x_challenge; we then batch it with g_i in Step 7.
    let batched_poly =
        DenseMultilinearExtension::from_evaluations_vec(num_vars.into(), batched_evals.clone());
    let batched_eval = y_f
        + y_js
            .iter()
            .enumerate()
            .map(|(j, &y_j)| hat_c_powers[j + 1] * y_j)
            .sum::<E::ScalarField>();
    let zeromorph_pp = ZeromorphProverKey::<E> {
        hiding_kzg_pp: pk.ck.clone(),
        open_offset: 0,
    };
    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let (batched_instance, q_hat_com, q_k_com) = Zeromorph::open_to_batched_instance(
        &zeromorph_pp,
        &batched_poly,
        &xs,
        batched_eval,
        Scalar(batched_randomness),
        rng,
        &mut trs,
    );
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("Zeromorph open_to_batched_instance", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 7 (spec): batch open Zeromorph reduced poly at x_challenge (eval 0) and g_i at x_i
    let g_i_evals_at_xi: Vec<E::ScalarField> = g_is
        .iter()
        .zip(xs.iter())
        .map(|(g_i_coeffs, x)| {
            let poly = DensePolynomial::from_coefficients_vec(g_i_coeffs.clone());
            poly.evaluate(x)
        })
        .collect();
    let y_g: E::ScalarField = g_i_evals_at_xi.iter().sum();

    let mut all_f_is = vec![batched_instance.f_coeffs];
    all_f_is.extend(g_is);
    let eval_points: Vec<E::ScalarField> =
        once(batched_instance.x).chain(xs.iter().copied()).collect();
    let mut all_evals = vec![E::ScalarField::ZERO]; // batched poly vanishes at x_challenge
    all_evals.extend(g_i_evals_at_xi.iter().copied());
    let mut all_rs = vec![batched_instance.rho];
    all_rs.extend(g_comm_randomnesses.iter().copied());
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative(
        "zk_pcs_open inputs (all_f_is, eval_points, all_evals, all_rs)",
        start.elapsed(),
    );

    // First poly (Zeromorph): eval 0 at x_challenge is revealed. Rest (g_i at x_i) stay hidden.
    let sets: Vec<EvaluationSet<E::ScalarField>> = eval_points
        .iter()
        .enumerate()
        .map(|(i, &pt)| {
            if i == 0 {
                EvaluationSet {
                    rev: vec![pt],
                    hid: vec![],
                }
            } else {
                EvaluationSet {
                    rev: vec![],
                    hid: vec![pt],
                }
            }
        })
        .collect();
    let polys: Vec<DensePolynomial<E::ScalarField>> = all_f_is
        .iter()
        .map(|coeffs| DensePolynomial::from_coefficients_vec(coeffs.clone()))
        .collect();
    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let opening = batch_open_generalized::<E, _, SumEvalHom<E::ScalarField>>(
        &srs,
        &sets,
        &polys,
        &all_rs,
        &SumEvalHom::<E::ScalarField>::default(),
        &mut trs,
        rng,
    );
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("batched opening", start.elapsed());

    let f_j_commitments: Vec<E::G1Affine> =
        f_j_comms_proj.iter().map(|g| g.into_affine()).collect();
    let g_i_commitments: Vec<E::G1Affine> = g_i_commitments_proj
        .iter()
        .map(|g| g.into_affine())
        .collect();

    Proof {
        blinding_poly_comm: comm_blinding_poly.map(|c| c.into_affine()),
        blinding_poly_proof: beta_sigma_proof,
        sumcheck_proof: sumcheck_proof.0,
        f_j_commitments,
        g_i_commitments,
        H_g,
        y_f,
        y_js, // Step 8: {y_j}_{1≤j≤ℓ} = f_j(x) at sumcheck point x
        y_g,
        zk_pcs_batch_proof: opening.proof,
        shplonked_eval_points: eval_points,
        zeromorph_q_hat_com: q_hat_com.0.into_affine(),
        zeromorph_q_k_com: q_k_com.iter().map(|c| c.0.into_affine()).collect(),
    }
}

/// Run sumcheck on (Σ c^j f_j(1-f_j)) Z_0 + α*g using aptos-crypto BooleanityEq LSB with masking.
/// Draws eq_point c_zc from trs, then runs BatchedSumcheck with BooleanityEqSumcheckProverLSB::new_with_masking.
#[allow(clippy::type_complexity)]
fn zkzc_send_polys<E: Pairing>(
    trs: &mut merlin::Transcript,
    g_is: Vec<Vec<E::ScalarField>>,
    num_vars: u8,
    ell: usize,
    c: E::ScalarField,
    alpha: E::ScalarField,
    hat_f_j_evals: &[Vec<E::ScalarField>],
    _timing: Option<&mut dyn FnMut(&str, std::time::Duration)>,
) -> (ClearSumcheckProof<E::ScalarField>, Vec<E::ScalarField>) {
    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = std::time::Instant::now();
    let c_zc = <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_point(trs, num_vars);
    let nv = num_vars as usize;
    #[cfg(feature = "range_proof_timing_multivariate")]
    if let Some(ref mut f) = _timing {
        f("zkzc_send_polys: eq_point c_zc", start.elapsed());
    }

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = std::time::Instant::now();
    // Build masking polynomial g(X) = sum_i g_i(X_i) from g_is (each g_i has 5 coeffs, degree 4)
    let univariate_polys: Vec<UniPoly<E::ScalarField>> = g_is
        .iter()
        .map(|coeffs| UniPoly::from_coeff(coeffs.clone()))
        .collect();
    let g = MaskingPolynomial::new(E::ScalarField::ZERO, univariate_polys);

    // MLE evals: one vec per f_j (BooleanityEq: sum_j c^j f_j(1-f_j) * eq_t * Z_0)
    let mle_evals: Vec<Vec<E::ScalarField>> =
        hat_f_j_evals[..ell].iter().map(|v| v.to_vec()).collect();
    let mut prover =
        BooleanityEqSumcheckProverLSB::new_with_masking(nv, mle_evals, c, c_zc, alpha, g);
    #[cfg(feature = "range_proof_timing_multivariate")]
    if let Some(ref mut f) = _timing {
        f(
            "zkzc_send_polys: BooleanityEqSumcheckProverLSB build",
            start.elapsed(),
        );
    }

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = std::time::Instant::now();
    let mut sumcheck_transcript = MerlinSumcheckTranscript::new(trs);
    let mut prover_acc = ProverOpeningAccumulator::new(0);
    let (proof, r_sumcheck, _initial_claim) = BatchedSumcheck::prove::<E::ScalarField, _>(
        vec![&mut prover],
        &mut prover_acc,
        &mut sumcheck_transcript,
    );
    #[cfg(feature = "range_proof_timing_multivariate")]
    if let Some(ref mut f) = _timing {
        f("zkzc_send_polys: BatchedSumcheck::prove", start.elapsed());
    }
    let challenges: Vec<E::ScalarField> = r_sumcheck;
    (proof, challenges)
}

// #[allow(non_snake_case)]
// fn prove<R: RngCore + CryptoRng>(
//     pk: &ProverKey<E>,
//     values: &[Self::Input],
//     ell: usize,
//     comm: &Self::Commitment,
//     rho: &Self::CommitmentRandomness,
//     rng: &mut R,
// ) -> Proof<E>
// {
//     // Step 1(a): Sample beta
//     let beta = sample_field_element(rng);

//     // Step 1(b): Commit to `beta \cdot eq_(1,..., 1)`, and prove knowledge of `beta`
//     let hom = two_term_msm::Homomorphism {
//             base_1: pk.lagr_g1.last().unwrap(),
//             base_2: pk.xi_1,
//         };
//     let rho = sample_field_element(rng);
//     let witness = two_term_msm::Witness {
//                 poly_randomness: Scalar(beta),
//                 hiding_kzg_randomness: Scalar(rho),
//             };
//     let blinding_poly_comm = hom.apply(&witness);
//     let sigma_proof = hom.prove(&witness, &blinding_poly_comm, &(), rng);

//     // // Step 1(b): commit to beta \cdot eq_(1,..., 1)
//     // let num_vars = (values.len() + 1).next_power_of_two().ilog2() as usize;
//     // let size = 1 << num_vars;
//     // let mut blinding_poly_values = vec![E::ScalarField::ZERO; size];
//     // blinding_poly_values[size - 1] = beta;
//     // let blinding_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, blinding_poly_values);
//     // let blinding_poly_comm = Zeromorph::commit(ck, &blinding_poly, rng);

//     // Step 3: Produce correlated randomness
//     let betas = correlated_randomness(rng, 2, ell.try_into().unwrap(), beta);

//     // Step 4: construct the hat_f_js
//     // This is copy-paste:
//     let bits: Vec<Vec<bool>> = values
//         .iter()
//         .map(|z_val| {
//             utils::scalar_to_bits_le::<E>(z_val)
//                 .into_iter()
//                 .take(ell)
//                 .collect::<Vec<_>>()
//         })
//         .collect();
//     // This is copy-paste:
//     let hat_f_j_evals_without_r: Vec<Vec<bool>> = (0..ell)
//         .map(|j| bits.iter().map(|row| row[j]).collect())
//         .collect(); // This is just transposing the bits matrix
//     let hat_f_j_evals: Vec<Vec<E::ScalarField>> = hat_f_j_evals_without_r
//         .iter()
//         .enumerate()
//         .map(|(j, col)| {
//             once(betas[j])
//                 .chain(col.iter().map(|&b| E::ScalarField::from(b)))
//                 .collect()
//         })
//         .collect();

//     let num_vars = (values.len() + 1).next_power_of_two().ilog2() as usize;
//     let hat_f_js: Vec<DenseMultilinearExtension::<E::ScalarField>> = hat_f_j_evals
//         .iter()
//         .map(|hat_f_j_eval| DenseMultilinearExtension::from_evaluations_vec(num_vars, hat_f_j_eval.clone()))
//         .collect();

//     // Step 5: Commit to the hat_f_j
//     let hat_f_j_comms: Vec<_> = hat_f_js
//         .iter()
//         .map(|hat_f_j| Zeromorph::commit(ck, hat_f_j, rng))
//         .collect();

//     // Step 6
//     let gammas = sample_field_elements(ell, rng);
//     // TODO: replace this with Fiat-Shamir?

//     // // Step 2(a):
//     // let poly = SparsePolynomial::from_coefficients_vec(
//     //     num_vars,
//     //     vec![
//     //         (sample_field_element(rng), SparseTerm::new(vec![])),
//     //         (sample_field_element(rng), SparseTerm::new(vec![(i, 1)])),
//     //         (sample_field_element(rng), SparseTerm::new(vec![(i, 2)])),
//     //     ],
//     // );

//     let g_is: Vec<_> = (0..num_vars)
//         .map(|i| {
//             SparsePolynomial::from_coefficients_vec(
//                 num_vars,
//                 vec![
//                     (sample_field_element(rng), SparseTerm::new(vec![])),
//                     (sample_field_element(rng), SparseTerm::new(vec![(i, 1)])),
//                     (sample_field_element(rng), SparseTerm::new(vec![(i, 2)])),
//                 ],
//             )
//         })
//         .collect();

//     // Step 2(b):
//     let g = g_is.iter().cloned().sum();

//     // // Step 2(c):
//     // let g_comm = Zeromorph::commit(ck, &g, rng);

//     // let mut G = E::ScalarField::ZERO;
//     // for i in 0..(1 << num_vars) {
//     //     // build the Boolean vector corresponding to i
//     //     let point: Vec<E::ScalarField> = (0..num_vars)
//     //         .map(|j| if (i >> j) & 1 == 1 {
//     //             E::ScalarField::ONE
//     //         } else {
//     //             E::ScalarField::ZERO
//     //         })
//     //         .collect();

//     //     G += g.evaluate(&point);
//     // }

// }
//

fn zksc_blind<E: Pairing, R: RngCore + CryptoRng>(
    last_msm_elt: E::G1Affine,
    xi_1: E::G1Affine,
    rng: &mut R,
) -> (
    E::ScalarField,
    E::G1,
    E::ScalarField,
    two_term_msm::Proof<E::G1>,
) {
    // Step 1: Sample `beta`
    let beta = sample_field_element(rng);

    // Step 2: Commit to `beta \cdot eq_(1,..., 1)` using a simplified version of Zeromorph
    let hom = two_term_msm::Homomorphism {
        base_1: last_msm_elt,
        base_2: xi_1,
    };
    let rho = sample_field_element(rng);
    let witness = two_term_msm::Witness {
        poly_randomness: Scalar(beta),
        hiding_kzg_randomness: Scalar(rho),
    };
    let blinding_poly_comm = hom.apply(&witness);

    // Step 3: Prove knowledge of `beta`
    let (sigma_proof, _) = hom.prove(&witness, blinding_poly_comm.clone(), &(), rng);

    (beta, blinding_poly_comm.0, rho, sigma_proof)
}

fn zksc_send_mask<E: Pairing, R: RngCore + CryptoRng>(
    srs: &Srs<E>,
    d: u8,
    num_vars: u8,
    rng: &mut R,
) -> (
    Vec<Vec<E::ScalarField>>,
    Vec<E::G1>,
    Vec<E::ScalarField>,
    E::ScalarField,
) {
    // Step (1): Sample the g_i
    let g_is: Vec<_> = (0..num_vars)
        .map(|_| sample_field_elements((d + 1).into(), rng))
        .collect();

    // Step (2): Commit
    let r_is = sample_field_elements(num_vars.into(), rng);
    let g_comm: Vec<E::G1> = g_is
        .iter()
        .zip(r_is.iter())
        .map(|(g_i, r_i)| {
            Shplonked::<E>::commit(
                srs,
                DensePolynomial::from_coefficients_vec(g_i.clone()),
                Some(*r_i),
            )
            .0
        })
        .collect();

    let mut sum_c = E::ScalarField::ZERO;
    let mut sum_b = E::ScalarField::ZERO;

    for g_i in g_is.clone() {
        sum_c += g_i[0];
        sum_b += g_i[1..].iter().copied().sum::<E::ScalarField>();
    }

    let two = E::ScalarField::from(2u64);
    let total_sum = two.pow([num_vars as u64]) * sum_c + two.pow([(num_vars - 1) as u64]) * sum_b;

    (g_is, g_comm, r_is, total_sum)
}

// /// Samples a specific kind of random polynomial `g`, then evaluates it at all points in {0,1}^num_vars and returns the polynomial, this sum and a commitment
// fn send_mask<E: Pairing, R: RngCore + CryptoRng>(ck: ZeromorphProverKey<E>, d: u8, num_vars: u8, rng: &mut R) -> (SparsePolynomial<E::ScalarField, SparseTerm>, E::G1, E::ScalarField) {

//     // Step (a): Sample the g_i
//     let g_is: Vec<_> = (0..num_vars)
//         .map(|i| {
//             SparsePolynomial::from_coefficients_vec(
//                 num_vars.into(),
//                 (0..=d)
//                     .map(|k| {
//                         let term = if k == 0 {
//                             // constant term
//                             SparseTerm::new(vec![])
//                         } else {
//                             SparseTerm::new(vec![(i.into(), k as usize)])
//                         };

//                         (sample_field_element(rng), term)
//                     })
//                     .collect())
//             })
//         .collect();

//     // Step (b): Sum them into one polynomial
//     let g = g_is.iter().cloned().sum();

//     // Step (c): Commit and compute the sum
//     let g_comm = univariate_hiding_kzg::commit(&ck, g, rng);

//     let mut sum = E::ScalarField::ZERO;

//     for i in 0..(1 << num_vars) {
//         // build the Boolean vector corresponding to i
//         let point: Vec<E::ScalarField> = (0..num_vars)
//             .map(|j| if (i >> j) & 1 == 1 { E::ScalarField::ONE } else { E::ScalarField::ZERO })
//             .collect();

//         sum += g.evaluate(&point);
//     }

//     (g, sum, comm)
// }

// pub mod blinding_check {
//     // TODO: maybe fixed_base_msms should become a folder and put its code inside mod.rs? Then put this mod inside of that folder?
//     use super::*;
//     use crate::sigma_protocol::{homomorphism::fixed_base_msms, traits::FirstProofItem};
//     use aptos_crypto::arkworks::{msm::IsMsmInput, random::UniformRand};
//     use aptos_crypto_derive::SigmaProtocolWitness;
//     use ark_ec::AffineRepr;
//     pub use sigma_protocol::homomorphism::TrivialShape as CodomainShape;
//     pub type Proof<C> = sigma_protocol::Proof<
//         <<C as CurveGroup>::Affine as AffineRepr>::ScalarField,
//         Homomorphism<C>,
//     >;

//     /// Represents a homomorphism with two base points over an elliptic curve group.
//     ///
//     /// This structure defines a map from two scalars to one group element:
//     /// `f(x1, x2) = base_1 * x1 + base_2 * x2`.
//     #[derive(CanonicalSerialize, Clone, Debug, PartialEq, Eq)]
//     pub struct Homomorphism<C: CurveGroup> {
//         pub base_1: C::Affine,
//         pub base_2: C::Affine,
//     }

//     #[derive(
//         SigmaProtocolWitness, CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq,
//     )]
//     pub struct Witness<F: PrimeField> {
//         pub poly_randomness: Scalar<F>,
//         pub hiding_kzg_randomness: Scalar<F>,
//     }

//     impl<C: CurveGroup> homomorphism::Trait for Homomorphism<C> {
//         type Codomain = CodomainShape<C>;
//         type Domain = Witness<C::ScalarField>;

//         fn apply(&self, input: &Self::Domain) -> Self::Codomain {
//             // Not doing `self.apply_msm(self.msm_terms(input))` because E::G1::msm is slower!
//             // `msm_terms()` is still useful for verification though: there the code will use it to produce an MSM
//             //  of size 2+2 (the latter two are for the first prover message A and the statement P)
//             CodomainShape(
//                 self.base_1 * input.poly_randomness.0 + self.base_2 * input.hiding_kzg_randomness.0,
//             )
//         }
//     }

//     impl<C: CurveGroup> fixed_base_msms::Trait for Homomorphism<C> {
//         type Base = C::Affine;
//         type CodomainShape<T>
//             = CodomainShape<T>
//         where
//             T: CanonicalSerialize + CanonicalDeserialize + Clone + Eq + Debug;
//         type MsmInput = MsmInput<C::Affine, C::ScalarField>;
//         type MsmOutput = C;
//         type Scalar = C::ScalarField;

//         fn msm_terms(&self, input: &Self::Domain) -> Self::CodomainShape<Self::MsmInput> {
//             let mut scalars = Vec::with_capacity(2);
//             scalars.push(input.poly_randomness.0);
//             scalars.push(input.hiding_kzg_randomness.0);

//             let mut bases = Vec::with_capacity(2);
//             bases.push(self.base_1);
//             bases.push(self.base_2);

//             CodomainShape(MsmInput { bases, scalars })
//         }

//         fn msm_eval(input: Self::MsmInput) -> Self::MsmOutput {
//             C::msm(input.bases(), input.scalars()).expect("MSM failed in TwoTermMSM")
//         }

//         fn batch_normalize(msm_output: Vec<Self::MsmOutput>) -> Vec<Self::Base> {
//             C::normalize_batch(&msm_output)
//         }
//     }

//     impl<C: CurveGroup> sigma_protocol::Trait<C> for Homomorphism<C> {
//         fn dst(&self) -> Vec<u8> {
//             b"DEKART_V2_SIGMA_PROTOCOL".to_vec()
//         }
//     }
// }

// mod ml_sumcheck {
//     /// Prover Message
//     #[derive(Clone, CanonicalSerialize)]
//     pub struct ProverMsg<F: Field> {
//         /// evaluations on P(0), P(1), P(2), ...
//         pub(crate) evaluations: Vec<F>,
//     }

//     /// Prover State for binary constraints with eq_t masking and g polynomial
//     pub struct ProverState<F: Field> {
//         /// sampled randomness given by the verifier
//         pub randomness: Vec<F>,
//         /// List of (coefficient, polynomial) pairs
//         pub constraints: Vec<(F, DenseMultilinearExtension<F>)>,
//         /// The eq_t point (original, never modified)
//         pub eq_point_original: Vec<F>,
//         /// Coefficient α for g term
//         pub alpha: F,
//         /// Random univariate polynomials g₁, ..., gₙ (coefficients)
//         pub g_polys: Vec<Vec<F>>,
//         /// Number of variables
//         pub num_vars: usize,
//         /// The current round number
//         pub round: usize,
//     }

// }

#[cfg(test)]
mod tests {
    use super::*;
    use aptos_crypto::arkworks::GroupGenerators;
    use ark_bn254::Bn254;
    use rand::thread_rng;

    #[test]
    fn test_prove_verify_simple() {
        type E = Bn254;
        let mut rng = thread_rng();
        let group_generators = GroupGenerators::default();

        // Setup: max_n = 4 so size = 8 (SRS supports degree-4 g), num_vars = 3; max_ell = 8.
        let max_n = 4;
        let max_ell = 8u8;
        let (pk, vk) = <Proof<E> as traits::BatchedRangeProof<E>>::setup(
            max_n,
            max_ell,
            group_generators,
            &mut rng,
        );

        // Four values so (4+1).next_power_of_two() = 8, num_vars = 3, matching vk.
        let values: Vec<ark_bn254::Fr> = vec![
            ark_bn254::Fr::from(0u64),
            ark_bn254::Fr::from(42u64),
            ark_bn254::Fr::from(1u64),
            ark_bn254::Fr::from(100u64),
        ];
        let n = values.len();

        let ck = <Proof<E> as traits::BatchedRangeProof<E>>::commitment_key_from_prover_key(&pk);
        let (comm, r) = <Proof<E> as traits::BatchedRangeProof<E>>::commit(&ck, &values, &mut rng);

        let proof = <Proof<E> as traits::BatchedRangeProof<E>>::prove(
            &pk,
            &values,
            max_ell,
            &comm.clone().into(),
            &r,
            &mut rng,
        );

        // Assert proof structure
        assert_eq!(
            proof.sumcheck_proof.compressed_polys.len(),
            3,
            "sumcheck rounds = num_vars"
        );
        assert_eq!(proof.f_j_commitments.len(), max_ell as usize);
        assert_eq!(
            proof.y_js.len(),
            max_ell as usize,
            "y_1..y_ell (y_f is separate)"
        );

        traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
            .expect("verification should succeed");
    }
}
