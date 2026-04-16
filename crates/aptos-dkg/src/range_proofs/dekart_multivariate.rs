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
        zeromorph::{replay_challenges, zeta_z_com, Zeromorph, ZeromorphProverKey, ZeromorphVerifierKey, ZeromorphCommitment, ZeromorphProof},
        EvaluationSet,
    },
    pvss::chunky::chunked_elgamal::correlated_randomness,
    range_proofs::{dekart_univariate_v2::two_term_msm, traits, PublicStatement},
    sigma_protocol::{
        homomorphism::{Trait as _, TrivialShape},
        Trait as _,
    },
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
        MerlinSumcheckTranscript, ProverOpeningAccumulator, UniPoly, VerifierOpeningAccumulator,
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
use std::time::Instant;
#[cfg(feature = "range_proof_timing_multivariate")]
use std::time::{Duration};
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
    /// Number of variables, needed for sumcheck (log2 of domain size)
    pub(crate) num_variables: usize,
    srs: Srs<E>,
}

#[allow(non_snake_case)]
#[derive(CanonicalSerialize, Clone, CanonicalDeserialize)]
pub struct Proof<E: Pairing> {
    /// Blinding commitment C_β
    pub blinding_poly_comm: E::G1Affine,
    /// Proof that C_β is of the form β·eq_0
    pub blinding_poly_proof: two_term_msm::Proof<E::G1>,
    pub f_j_comms: Vec<E::G1Affine>,
    pub g_i_comms: Vec<E::G1Affine>,
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
        if let Some(c) = self.f_j_comms.first_mut() {
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
                "{:>10.2} ms  ({:>10.2} ms cum.)  [dekart_multivariate verify] {}",
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
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_blinding_poly_commitment(
            &mut trs, &self.blinding_poly_comm,
        );
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_sigma_proof(
            &mut trs, &self.blinding_poly_proof,
        );
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("transcript init (vk + public statement)", start.elapsed());

        // Step 2: Verify the blinding commitment
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();

        let hom = two_term_msm::Homomorphism {
                base_1: vk.vk_hkzg.group_generators.g1,
                base_2: vk.xi_1,
        };
        hom.verify(
            &TrivialShape(self.blinding_poly_comm.into()),
            &self.blinding_poly_proof,
            &(),
            rng,
        )?;


        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("blinding two_term_msm verify", start.elapsed());

        // Step 3a–3d: Append commitments and draw challenges
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();

        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_f_j_commitments(
            &mut trs,
            &self.f_j_comms,
        );
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_g_i_commitments(
            &mut trs,
            &self.g_i_comms,
        );
        <merlin::Transcript as RangeProof<E, Proof<E>>>::append_hypercube_sum(&mut trs, &self.H_g);

        let c_s =
            <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_point(&mut trs, ell);
        let alpha: E::ScalarField =
            <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_nonzero_scalar(&mut trs);
        let c_zc = <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_point(
            &mut trs,
            num_vars as u8,
        );
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative(
            "append commitments + challenges (c_s, alpha, c_zc)",
            start.elapsed(),
        );

        // Step 3e: Sumcheck verify (aptos-crypto BooleanityEq LSB with alpha*g masking)
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let claimed_sum = alpha * self.H_g;
        let alpha_y_g = alpha * self.y_g;
        let verifier = BooleanityEqSumcheckVerifierLSBWithOpenings::new_with_alpha_y_g(
            num_vars,
            c_s.clone(),
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
            for (j, &y_j) in self.y_js.iter().take(ell as usize).enumerate() {
                sum_c_j += c_s[j] * y_j * (E::ScalarField::ONE - y_j);
            }
        let  subclaim_expected_eval =   sum_c_j * eq_c_zc_at_x * z_0_at_x + alpha * self.y_g;

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

        let lhs_4a = sum_c_j * eq_c_zc_at_x * Z_0_at_x + alpha * self.y_g;
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

        // Step 5a: Add y_f and {y_j}_{1≤j≤ℓ} to the Fiat–Shamir transcript.
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        trs.append_evaluation_points(&[self.y_f]);
        for &y_j in self.y_js.iter().take(ell as usize) {
            trs.append_evaluation_points(&[y_j]);
        }
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("transcript y_f, y_js", start.elapsed());

        // Step 5b: Challenge hat_c.
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let hat_c: E::ScalarField =
            <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_scalar(&mut trs);
        let hat_c_powers = powers(hat_c, ell as usize + 1);
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("hat_c + powers", start.elapsed());

        // Step 5c: Replay mPCS.ReduceToUnivariate (Zeromorph) to get x_challenge and form zeta_z_com.
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let (y_challenge, x_challenge, z_challenge) =
            replay_challenges::<E>(&mut trs, &self.zeromorph_q_k_com, &self.zeromorph_q_hat_com);
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("replay_challenges", start.elapsed());

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

        // Now form the MSM input corresponding to batching the Zeromorph openings (no MSM evaluation yet).
        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let mut combined_bases = vec![comm.0.into_affine(), self.blinding_poly_comm];
        let mut combined_scalars = vec![E::ScalarField::ONE, E::ScalarField::ONE];
        combined_bases.extend(self.f_j_comms.iter().copied());
        combined_scalars.extend(hat_c_powers.iter().skip(1).copied());
        let combined_comm =
            MsmInput::new(combined_bases, combined_scalars).expect("combined commitment MSM input");
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("combined_comm (MsmInput)", start.elapsed());

        #[cfg(feature = "range_proof_timing_multivariate")]
        let start = Instant::now();
        let point_reversed: Vec<E::ScalarField> = x.iter().rev().cloned().collect();
        let zeromorph_msm = zeta_z_com::<E>(
            self.zeromorph_q_hat_com,
            combined_comm,
            vk.vk_hkzg.group_generators.g1,
            &self.zeromorph_q_k_com,
            y_challenge,
            x_challenge,
            z_challenge,
            &point_reversed,
            batched_eval,
        );
        #[cfg(feature = "range_proof_timing_multivariate")]
        print_cumulative("zeta_z_com", start.elapsed());

        let g_commitment_msms: Vec<MsmInput<E::G1Affine, E::ScalarField>> = self
            .g_i_comms
            .iter()
            .map(|&affine| {
                MsmInput::new(vec![affine], vec![E::ScalarField::ONE]).expect("single term")
            })
            .collect();
        let commitment_msms: Vec<MsmInput<E::G1Affine, E::ScalarField>> =
            once(zeromorph_msm).chain(g_commitment_msms).collect();

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
        let comm_conv = TrivialShape(comm.0.into_group()); // TODO: hacky, remove etc
        prove_impl(pk, values, ell, &comm_conv, rho, rng)
    }
}

/// Prover
#[allow(non_snake_case)]
pub fn prove_impl<E: Pairing, R: RngCore + CryptoRng>(
    pk: &ProverKey<E>,
    values: &[E::ScalarField], // Might make sense to start this array with β
    ell: u8,
    comm: &univariate_hiding_kzg::Commitment<E>,
    rho: &univariate_hiding_kzg::CommitmentRandomness<E::ScalarField>,
    rng: &mut R,
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

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let mut trs = merlin::Transcript::new(b"dekart_multivariate");
    let tau_powers = match &pk.ck.msm_basis {
        SrsBasis::PowersOfTau { tau_powers } => tau_powers,
        _ => panic!("Expected PowersOfTau SRS"),
    };
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_vk(&mut trs, &pk.vk);
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_public_statement(
        &mut trs,
        PublicStatement {
            n: values.len(), // So this is a power of two, minus one
            ell,
            comm: comm.clone(),
        },
    );
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("transcript init (vk + public statement)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();

    let (beta, comm_blinding_poly, comm_blinding_poly_rand, beta_sigma_proof) = {
        let g1_generator = pk.vk.vk_hkzg.group_generators.g1;
        let (b, c, r, proof): (
            E::ScalarField,
            E::G1,
            E::ScalarField,
            two_term_msm::Proof<E::G1>,
        ) = zksc_blind::<E, _>(g1_generator, pk.ck.xi_1, rng);
        let c_affine = c.into_affine();

    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_blinding_poly_commitment(
        &mut trs, &c_affine,
    );
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_sigma_proof(
        &mut trs, &proof,
    );
        (b, c, r, proof)
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
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("hat_f_j commitments (hom.apply loop)", start.elapsed());


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

    let n_f = f_j_comms_proj.len();
    let mut combined = f_j_comms_proj;
    combined.extend(&g_i_commitments_proj);
    let normalized = E::G1::normalize_batch(&combined);
    let f_j_comms = normalized[..n_f].to_vec();
    let g_i_comms = normalized[n_f..].to_vec();

    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("zksc_send_mask (g_is, g_comm, G)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 4b: Add {C_{f_j}}, {C_{g_i}} and H_g to the Fiat–Shamir transcript.
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_f_j_commitments(&mut trs, &f_j_comms);
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_g_i_commitments(&mut trs, &g_i_comms);
    <merlin::Transcript as RangeProof<E, Proof<E>>>::append_hypercube_sum(&mut trs, &H_g);
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("transcript append comm_blinding_poly + f_j_comms + g_comm + H_g", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let mut f_evals = vec![E::ScalarField::ZERO; size];
    f_evals[0] = beta;
    for (i, &v) in values.iter().enumerate() {
        f_evals[i + 1] = v;
    }
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("f_evals construction", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 5a–5c: Verifier challenges c, alpha; eq_point t; run sumcheck on transcript with linear term (f - sum 2^{j-1} f_j) + sum c^j f_j(f_j-1)
    let c_s =
        <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_point(&mut trs, ell);
    let alpha: E::ScalarField =
        <merlin::Transcript as RangeProof<E, Proof<E>>>::challenge_nonzero_scalar(&mut trs);
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("transcript challenges (c, alpha)", start.elapsed());

    // TODO: define hat(f) hier ipv in zkzc_send_polys()
    // mytodo: check sum-check implementation (including fiat-shamir implementation)
    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    let sumcheck_proof = zkzc_send_polys::<E>(
        &mut trs,
        g_is.clone(),
        num_vars,
        ell as usize,
        c_s,
        alpha,
        &f_j_evals,
        #[cfg(feature = "range_proof_timing_multivariate")]
        Some(&mut print_cumulative),
        #[cfg(not(feature = "range_proof_timing_multivariate"))]
        None,
    );
    #[cfg(feature = "range_proof_timing_multivariate")]
    print_cumulative("zkzc_send_polys (sumcheck total)", start.elapsed());

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
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
        "xs + y_f/y_js eval + transcript append + hat_c",
        start.elapsed(),
    );

    #[cfg(feature = "range_proof_timing_multivariate")]
    let start = Instant::now();
    // Step 6e:
    // Batched polynomial f̂ = f + blinding_poly + sum_j hat_c^j f_j (coefficient form for univariate opening)
    let hat_c_powers = powers(hat_c, ell as usize + 1);
    let mut batched_evals = f_evals.clone();
    for j in 0..ell as usize {
        let cj = hat_c_powers[j + 1];
        for (i, b) in batched_evals.iter_mut().enumerate() {
            *b += cj * f_j_evals[j][i];
        }
    }

    let mut batched_randomness = rho.0 + comm_blinding_poly_rand;
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
    print_cumulative(
        "batched_evals + batched_poly + batched_randomness + zeromorph_pp",
        start.elapsed(),
    );
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

    Proof {
        blinding_poly_comm: comm_blinding_poly.into_affine(),
        blinding_poly_proof: beta_sigma_proof,
        sumcheck_proof: sumcheck_proof.0,
        f_j_comms,
        g_i_comms,
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
    c_s: Vec<E::ScalarField>,
    alpha: E::ScalarField,
    hat_f_j_evals: &[Vec<E::ScalarField>],
    mut _timing: Option<&mut dyn FnMut(&str, std::time::Duration)>,
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
        BooleanityEqSumcheckProverLSB::new_with_masking(nv, mle_evals, c_s, c_zc, alpha, g);
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

#[cfg(test)]
mod tests {
    use super::*;
    use aptos_crypto::arkworks::GroupGenerators;
    use ark_bn254::Bn254;
    use ark_ff::{UniformRand, PrimeField};  // Add PrimeField
    use rand::thread_rng;

    /// Helper function to generate random field elements in a specified range [0, max_value)
    fn generate_random_values_in_range<R: RngCore + CryptoRng>(
        n: usize,
        max_value: u64,
        rng: &mut R,
    ) -> Vec<ark_bn254::Fr> {
        (0..n)
            .map(|_| {
                // Use sample_field_element and then modulo to get into range
                let random_val = sample_field_element::<ark_bn254::Fr, _>(rng);
                let val_u64 = random_val.into_bigint().0[0] % max_value;
                ark_bn254::Fr::from(val_u64)
            })
            .collect()
    }

    /// Helper function to generate completely random field elements (no range constraint)
    fn generate_random_field_elements<R: RngCore + CryptoRng>(
        n: usize,
        rng: &mut R,
    ) -> Vec<ark_bn254::Fr> {
        (0..n)
            .map(|_| sample_field_element::<ark_bn254::Fr, _>(rng))
            .collect()
    }

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
        assert_eq!(proof.f_j_comms.len(), max_ell as usize);
        assert_eq!(
            proof.y_js.len(),
            max_ell as usize,
            "y_1..y_ell (y_f is separate)"
        );

        traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
            .expect("verification should succeed");
    }

    #[test]
    fn test_prove_verify_random_values_in_range() {
        type E = Bn254;
        let mut rng = thread_rng();
        let group_generators = GroupGenerators::default();

        // Parameters
        let n = 7; // Number of values
        let max_value = 1000u64; // Values will be in [0, 1000)
        let max_ell = 16u8; // Bit length

        // Setup with sufficient capacity
        let max_n = n;
        let (pk, vk) = <Proof<E> as traits::BatchedRangeProof<E>>::setup(
            max_n,
            max_ell,
            group_generators,
            &mut rng,
        );

        // Generate random values in range
        let values = generate_random_values_in_range(n, max_value, &mut rng);
        println!("Testing with {} random values in range [0, {})", n, max_value);
        println!("Values: {:?}", values.iter().map(|v| {
            v.into_bigint().0[0]
        }).collect::<Vec<_>>());

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

        traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
            .expect("verification should succeed for random values in range");
    }

    #[test]
    fn test_prove_verify_parameterized() {
        type E = Bn254;
        let mut rng = thread_rng();
        let group_generators = GroupGenerators::default();

        // Test with different parameters
        let test_cases: Vec<(usize, u64, u8)> = vec![
            (3, 256u64, 8u8),      // 3 values, range [0, 256), 8 bits
            (7, 1024u64, 16u8),    // 7 values, range [0, 1024), 16 bits
            (1000, 65536u64, 16u8),  // 15 values, range [0, 65536), 16 bits
            (1, 100u64, 8u8),      // Edge case: single value
        ];

        for (n, max_value, max_ell) in test_cases {
            println!("\n=== Testing: n={}, max_value={}, max_ell={} ===", n, max_value, max_ell);

            // Calculate required SRS size: need to support both value commitment and masking polys
            let size: usize = (n + 1).next_power_of_two();
            let num_vars = size.ilog2() as usize;
            // Each g_i has degree 4, and we have num_vars of them
            let max_n = n.max(num_vars * 5);  // Ensure we have enough for masking polynomials

            let (pk, vk) = <Proof<E> as traits::BatchedRangeProof<E>>::setup(
                max_n,
                max_ell,
                group_generators.clone(),
                &mut rng,
            );

            let values = generate_random_values_in_range(n, max_value, &mut rng);

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

            // Verify proof structure
            let expected_num_vars = (n + 1).next_power_of_two().ilog2() as usize;
            assert_eq!(
                proof.sumcheck_proof.compressed_polys.len(),
                expected_num_vars,
                "sumcheck rounds = num_vars for n={}",
                n
            );
            assert_eq!(proof.f_j_comms.len(), max_ell as usize);
            assert_eq!(proof.y_js.len(), max_ell as usize);

            traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
                .expect(&format!("verification should succeed for n={}, max_value={}, max_ell={}",
                    n, max_value, max_ell));

            println!("✓ Test passed");
        }
    }

    /// Customizable test function that can be called with specific parameters
    fn run_custom_test(n: usize, max_value: u64, max_ell: u8) {
        type E = Bn254;
        let mut rng = thread_rng();
        let group_generators = GroupGenerators::default();

        println!("\n=== Custom test: n={}, range=[0, {}), ell={} ===", n, max_value, max_ell);

        let size: usize = (n + 1).next_power_of_two();  // Add type annotation
        let num_vars = size.ilog2() as usize;
        let max_n = n.max(num_vars * 5);

        let (pk, vk) = <Proof<E> as traits::BatchedRangeProof<E>>::setup(
            max_n,
            max_ell,
            group_generators,
            &mut rng,
        );

        let values = generate_random_values_in_range(n, max_value, &mut rng);
        println!("Generated values: {:?}",
            values.iter().map(|v| v.into_bigint().0[0]).collect::<Vec<_>>()
        );

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

        // Verify proof structure
        let expected_num_vars: usize = (n + 1).next_power_of_two().ilog2() as usize;  // Add type annotation here too
        assert_eq!(
            proof.sumcheck_proof.compressed_polys.len(),
            expected_num_vars,
            "sumcheck rounds = num_vars"
        );
        assert_eq!(proof.f_j_comms.len(), max_ell as usize);
        assert_eq!(proof.y_js.len(), max_ell as usize);

        traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
            .expect("custom test verification should succeed");

        println!("✓ Custom test passed");
    }

    #[test]
    fn test_prove_verify_edge_cases() {
        type E = Bn254;
        let mut rng = thread_rng();
        let group_generators = GroupGenerators::default();

        // Test with all zeros
        {
            println!("\n=== Testing: all zeros ===");
            let n = 4;
            let max_ell = 8u8;
            let (pk, vk) = <Proof<E> as traits::BatchedRangeProof<E>>::setup(
                n,
                max_ell,
                group_generators.clone(),
                &mut rng,
            );

            let values = vec![ark_bn254::Fr::from(0u64); n];
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

            traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
                .expect("verification should succeed for all zeros");
            println!("✓ All zeros test passed");
        }

        // Test with maximum values for bit length
        {
            println!("\n=== Testing: maximum values ===");
            let n = 4;
            let max_ell = 8u8;
            let (pk, vk) = <Proof<E> as traits::BatchedRangeProof<E>>::setup(
                n,
                max_ell,
                group_generators.clone(),
                &mut rng,
            );

            let max_val = (1u64 << max_ell) - 1; // 2^ell - 1
            let values = vec![ark_bn254::Fr::from(max_val); n];

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

            traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
                .expect("verification should succeed for maximum values");
            println!("✓ Maximum values test passed");
        }

        // Test with powers of two
        {
            println!("\n=== Testing: powers of two ===");
            let n = 5;
            let max_ell = 16u8;
            let (pk, vk) = <Proof<E> as traits::BatchedRangeProof<E>>::setup(
                n,
                max_ell,
                group_generators.clone(),
                &mut rng,
            );

            let values: Vec<ark_bn254::Fr> = (0..n)
                .map(|i| ark_bn254::Fr::from(1u64 << i))
                .collect();

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

            traits::BatchedRangeProof::<E>::verify(&proof, &vk, n, max_ell, &comm, &mut rng)
                .expect("verification should succeed for powers of two");
            println!("✓ Powers of two test passed");
        }
    }


}
