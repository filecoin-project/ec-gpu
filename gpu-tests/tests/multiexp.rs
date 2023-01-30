#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::sync::Arc;
use std::time::Instant;

use blstrs::Bls12;
use ec_gpu::GpuName;
use ec_gpu_gen::multiexp_cpu::{multiexp_cpu, FullDensity, QueryDensity, SourceBuilder};
use ec_gpu_gen::{
    multiexp::{MultiexpKernel, SingleMultiexpKernel as SingleMultiexpKernelZcash},
    program,
    rust_gpu_tools::Device,
    threadpool::Worker,
    EcError,
};
use ff::{Field, PrimeField};
use group::Curve;
use group::{prime::PrimeCurveAffine, Group};
use log::debug;
use once_cell::sync::Lazy;
use pairing::Engine;

use ark_bls12_381::{
    g1::{Config as G1Config, BETA, G1_GENERATOR_X, G1_GENERATOR_Y},
    g2::Config as G2Config,
    Fq, Fq12Config, Fq2Config, Fq6Config, Fr,
};
use ark_ec::bls12::{Bls12 as Bls12Arkworks, Bls12Config, TwistType};
use ark_ec::models::CurveConfig;
use ark_ec::pairing::Pairing as PairingArkworks;
use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
use ark_ff::{BigInteger, BigInteger384, Field as _, One, PrimeField as _};
use ark_serialize::{CanonicalSerialize, Compress, SerializationError};
use group::GroupEncoding;
use std::ops::Neg;
use std::sync::Mutex;

static ARK_KERNEL_MULTIEXP: Lazy<Mutex<SingleMultiexpKernel<'static, G1Affine>>> =
    Lazy::new(|| {
        let devices = Device::all();
        let device = devices[0];
        let program = crate::program!(device).expect("Cannot create program!");
        let kernel = SingleMultiexpKernel::<<Bls12_381 as PairingArkworks>::G1Affine>::create(
            program, &device, None,
        )
        .expect("Cannot initialize kernel!");
        Mutex::new(kernel)
    });

#[derive(Clone, Default, PartialEq, Eq)]
struct G1ConfigGpu;

impl CurveConfig for G1ConfigGpu {
    type BaseField = <G1Config as CurveConfig>::BaseField;
    type ScalarField = <G1Config as CurveConfig>::ScalarField;

    /// COFACTOR = (x - 1)^2 / 3  = 76329603384216526031706109802092473003
    const COFACTOR: &'static [u64] = G1Config::COFACTOR;

    /// COFACTOR_INV = COFACTOR^{-1} mod r
    /// = 52435875175126190458656871551744051925719901746859129887267498875565241663483
    const COFACTOR_INV: ark_bls12_381::Fr = G1Config::COFACTOR_INV;
}

impl SWCurveConfig for G1ConfigGpu {
    /// COEFF_A = 0
    const COEFF_A: Fq = G1Config::COEFF_A;

    /// COEFF_B = 4
    const COEFF_B: Fq = G1Config::COEFF_B;

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: G1Affine = G1Affine::new_unchecked(G1_GENERATOR_X, G1_GENERATOR_Y);

    #[inline(always)]
    fn mul_by_a(a: Self::BaseField) -> Self::BaseField {
        G1Config::mul_by_a(a)
    }

    #[inline]
    fn is_in_correct_subgroup_assuming_on_curve(p: &G1Affine) -> bool {
        // Algorithm from Section 6 of https://eprint.iacr.org/2021/1130.
        //
        // Check that endomorphism_p(P) == -[X^2]P

        // An early-out optimization described in Section 6.
        // If uP == P but P != point of infinity, then the point is not in the right
        // subgroup.
        let x_times_p = p.mul_bigint(Config::X);
        if x_times_p.eq(p) && !p.infinity {
            return false;
        }

        let minus_x_squared_times_p = x_times_p.mul_bigint(Config::X).neg();
        let endomorphism_p = endomorphism(p);
        minus_x_squared_times_p.eq(&endomorphism_p)
    }

    #[inline]
    fn clear_cofactor(p: &G1Affine) -> G1Affine {
        // Using the effective cofactor, as explained in
        // Section 5 of https://eprint.iacr.org/2019/403.pdf.
        //
        // It is enough to multiply by (1 - x), instead of (x - 1)^2 / 3
        let h_eff = one_minus_x().into_bigint();
        Self::mul_affine(&p, h_eff.as_ref()).into()
    }

    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Affine<Self>, ark_serialize::SerializationError> {
        let p = if compress == ark_serialize::Compress::Yes {
            read_g1_compressed(&mut reader)?
        } else {
            read_g1_uncompressed(&mut reader)?
        };

        if validate == ark_serialize::Validate::Yes && !p.is_in_correct_subgroup_assuming_on_curve()
        {
            return Err(SerializationError::InvalidData);
        }
        Ok(p)
    }

    fn serialize_with_mode<W: ark_serialize::Write>(
        item: &Affine<Self>,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        let encoding = EncodingFlags {
            is_compressed: compress == ark_serialize::Compress::Yes,
            is_infinity: item.is_zero(),
            is_lexographically_largest: item.y > -item.y,
        };
        let mut p = *item;
        if encoding.is_infinity {
            p = G1Affine::zero();
        }
        // need to access the field struct `x` directly, otherwise we get None from xy()
        // method
        let x_bytes = serialize_fq(p.x);
        if encoding.is_compressed {
            let mut bytes: [u8; G1_SERIALIZED_SIZE] = x_bytes;

            encoding.encode_flags(&mut bytes);
            writer.write_all(&bytes)?;
        } else {
            let mut bytes = [0u8; 2 * G1_SERIALIZED_SIZE];
            bytes[0..G1_SERIALIZED_SIZE].copy_from_slice(&x_bytes[..]);
            bytes[G1_SERIALIZED_SIZE..].copy_from_slice(&serialize_fq(p.y)[..]);

            encoding.encode_flags(&mut bytes);
            writer.write_all(&bytes)?;
        };

        Ok(())
    }

    fn serialized_size(compress: Compress) -> usize {
        G1Config::serialized_size(compress)
    }

    fn msm(bases: &[G1Affine], scalars: &[Self::ScalarField]) -> Result<G1Projective, usize> {
        //debug!("vmx: starting scalar conversion");
        let bigints = scalars
            .iter()
            .map(|value| value.into_bigint())
            .collect::<Vec<_>>();
        //debug!("vmx: starting scalar conversion: done");
        let gpu = ARK_KERNEL_MULTIEXP
            .lock()
            .unwrap()
            .multiexp(&bases, &bigints[..])
            .unwrap();
        Ok(gpu)
    }
}

fn one_minus_x() -> Fr {
    const X: Fr = Fr::from_sign_and_limbs(!Config::X_IS_NEGATIVE, Config::X);
    Fr::one() - X
}

fn endomorphism(p: &G1Affine) -> G1Affine {
    // Endomorphism of the points on the curve.
    // endomorphism_p(x,y) = (BETA * x, y)
    // where BETA is a non-trivial cubic root of unity in Fq.
    let mut res = (*p).clone();
    res.x *= BETA;
    res
}

pub const G1_SERIALIZED_SIZE: usize = 48;
pub const G2_SERIALIZED_SIZE: usize = 96;

pub struct EncodingFlags {
    pub is_compressed: bool,
    pub is_infinity: bool,
    pub is_lexographically_largest: bool,
}

impl EncodingFlags {
    pub fn get_flags(bytes: &[u8]) -> Self {
        let compression_flag_set = (bytes[0] >> 7) & 1;
        let infinity_flag_set = (bytes[0] >> 6) & 1;
        let sort_flag_set = (bytes[0] >> 5) & 1;

        Self {
            is_compressed: compression_flag_set == 1,
            is_infinity: infinity_flag_set == 1,
            is_lexographically_largest: sort_flag_set == 1,
        }
    }

    pub fn encode_flags(&self, bytes: &mut [u8]) {
        if self.is_compressed {
            bytes[0] |= 1 << 7;
        }

        if self.is_infinity {
            bytes[0] |= 1 << 6;
        }

        if self.is_compressed && !self.is_infinity && self.is_lexographically_largest {
            bytes[0] |= 1 << 5;
        }
    }
}

fn read_g1_compressed<R: ark_serialize::Read>(
    mut reader: R,
) -> Result<G1Affine, ark_serialize::SerializationError> {
    let mut bytes = [0u8; G1_SERIALIZED_SIZE];
    reader
        .read_exact(&mut bytes)
        .ok()
        .ok_or(SerializationError::InvalidData)?;

    // Obtain the three flags from the start of the byte sequence
    let flags = EncodingFlags::get_flags(&bytes[..]);

    // we expect to be deserializing a compressed point
    if !flags.is_compressed {
        return Err(SerializationError::UnexpectedFlags);
    }

    if flags.is_infinity {
        return Ok(G1Affine::zero());
    }

    // Attempt to obtain the x-coordinate
    let x = read_fq_with_offset(&bytes, 0, true)?;

    let p = G1Affine::get_point_from_x_unchecked(x, flags.is_lexographically_largest)
        .ok_or(SerializationError::InvalidData)?;

    Ok(p)
}

pub(crate) fn read_g1_uncompressed<R: ark_serialize::Read>(
    mut reader: R,
) -> Result<G1Affine, ark_serialize::SerializationError> {
    let mut bytes = [0u8; 2 * G1_SERIALIZED_SIZE];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| SerializationError::InvalidData)?;

    // Obtain the three flags from the start of the byte sequence
    let flags = EncodingFlags::get_flags(&bytes[..]);

    // we expect to be deserializing an uncompressed point
    if flags.is_compressed {
        return Err(SerializationError::UnexpectedFlags);
    }

    if flags.is_infinity {
        return Ok(G1Affine::zero());
    }

    // Attempt to obtain the x-coordinate
    let x = read_fq_with_offset(&bytes, 0, true)?;
    // Attempt to obtain the y-coordinate
    let y = read_fq_with_offset(&bytes, 1, false)?;

    let p = G1Affine::new_unchecked(x, y);
    Ok(p)
}

pub(crate) fn deserialize_fq(bytes: [u8; 48]) -> Option<Fq> {
    let mut tmp = BigInteger384::new([0, 0, 0, 0, 0, 0]);

    // Note: The following unwraps are if the compiler cannot convert
    // the byte slice into [u8;8], we know this is infallible since we
    // are providing the indices at compile time and bytes has a fixed size
    tmp.0[5] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap());
    tmp.0[4] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap());
    tmp.0[3] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap());
    tmp.0[2] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap());
    tmp.0[1] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[32..40]).unwrap());
    tmp.0[0] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[40..48]).unwrap());

    Fq::from_bigint(tmp)
}

pub(crate) fn serialize_fq(field: Fq) -> [u8; 48] {
    let mut result = [0u8; 48];

    let rep = field.into_bigint();

    result[0..8].copy_from_slice(&rep.0[5].to_be_bytes());
    result[8..16].copy_from_slice(&rep.0[4].to_be_bytes());
    result[16..24].copy_from_slice(&rep.0[3].to_be_bytes());
    result[24..32].copy_from_slice(&rep.0[2].to_be_bytes());
    result[32..40].copy_from_slice(&rep.0[1].to_be_bytes());
    result[40..48].copy_from_slice(&rep.0[0].to_be_bytes());

    result
}

pub(crate) fn read_fq_with_offset(
    bytes: &[u8],
    offset: usize,
    mask: bool,
) -> Result<Fq, ark_serialize::SerializationError> {
    let mut tmp = [0; G1_SERIALIZED_SIZE];
    // read `G1_SERIALIZED_SIZE` bytes
    tmp.copy_from_slice(&bytes[offset * G1_SERIALIZED_SIZE..G1_SERIALIZED_SIZE * (offset + 1)]);

    if mask {
        // Mask away the flag bits
        tmp[0] &= 0b0001_1111;
    }
    deserialize_fq(tmp).ok_or(SerializationError::InvalidData)
}

struct Config;

impl Bls12Config for Config {
    const X: &'static [u64] = &[0xd201000000010000];
    const X_IS_NEGATIVE: bool = true;
    const TWIST_TYPE: TwistType = TwistType::M;
    type Fp = Fq;
    type Fp2Config = Fq2Config;
    type Fp6Config = Fq6Config;
    type Fp12Config = Fq12Config;
    type G1Config = G1ConfigGpu;
    type G2Config = G2Config;
}

type Bls12_381 = Bls12Arkworks<Config>;

type G1Affine = ark_ec::bls12::G1Affine<Config>;
type G1Projective = ark_ec::bls12::G1Projective<Config>;

fn multiexp_gpu<Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut MultiexpKernel<G>,
) -> Result<G::Curve, EcError>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine + GpuName,
    S: SourceBuilder<G>,
{
    let exps = density_map.as_ref().generate_exps::<G::Scalar>(exponents);
    let (bss, skip) = bases.get();
    kern.multiexp(pool, bss, exps, skip).map_err(Into::into)
}

#[test]
fn gpu_multiexp_consistency() {
    fil_logger::maybe_init();
    const MAX_LOG_D: usize = 16;
    const START_LOG_D: usize = 10;
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| crate::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = MultiexpKernel::<<Bls12 as Engine>::G1Affine>::create(programs, &devices)
        .expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    let mut bases = (0..(1 << START_LOG_D))
        .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
        .collect::<Vec<_>>();

    for log_d in START_LOG_D..=MAX_LOG_D {
        let g = Arc::new(bases.clone());

        let samples = 1 << log_d;
        println!("Testing Multiexp for {} elements...", samples);

        let v = Arc::new(
            (0..samples)
                .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
                .collect::<Vec<_>>(),
        );

        let mut now = Instant::now();
        let gpu = multiexp_gpu(&pool, (g.clone(), 0), FullDensity, v.clone(), &mut kern).unwrap();
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        let cpu = multiexp_cpu(&pool, (g.clone(), 0), FullDensity, v.clone())
            .wait()
            .unwrap();
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(cpu, gpu);

        println!("============================");

        bases = [bases.clone(), bases.clone()].concat();
    }
}

use std::ops::AddAssign;

use ark_ec::{AffineRepr, CurveGroup, Group as _};
use ark_ff::Zero;
use ec_gpu_gen::{
    multiexp::{div_ceil, work_units},
    rust_gpu_tools::{program_closures, Program},
    EcResult,
};

/// On the GPU, the exponents are split into windows, this is the maximum number of such windows.
const MAX_WINDOW_SIZE: usize = 10;
/// In CUDA this is the number of blocks per grid (grid size).
const LOCAL_WORK_SIZE: usize = 128;
/// Let 20% of GPU memory be free, this is an arbitrary value.
const MEMORY_PADDING: f64 = 0.2f64;
/// The Nvidia Ampere architecture is compute capability major version 8.
const AMPERE: u32 = 8;

/// Multiexp kernel for a single GPU.
pub struct SingleMultiexpKernel<'a, G>
where
    G: AffineRepr,
{
    program: Program,
    /// The number of exponentiations the GPU can handle in a single execution of the kernel.
    n: usize,
    /// The number of units the work is split into. It will results in this amount of threads on
    /// the GPU.
    work_units: usize,
    /// An optional function which will be called at places where it is possible to abort the
    /// multiexp calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,

    _phantom: std::marker::PhantomData<G>,
}

/// Calculates the maximum number of terms that can be put onto the GPU memory.
fn calc_chunk_size<G>(mem: u64, work_units: usize) -> usize
where
    G: AffineRepr,
    G::ScalarField: ark_ff::PrimeField,
{
    let aff_size = std::mem::size_of::<G>();
    let exp_size = exp_size::<G::ScalarField>();
    let proj_size = std::mem::size_of::<G::Group>();

    // Leave `MEMORY_PADDING` percent of the memory free.
    let max_memory = ((mem as f64) * (1f64 - MEMORY_PADDING)) as usize;
    // The amount of memory (in bytes) of a single term.
    let term_size = aff_size + exp_size;
    // The number of buckets needed for one work unit
    let max_buckets_per_work_unit = 1 << MAX_WINDOW_SIZE;
    // The amount of memory (in bytes) we need for the intermediate steps (buckets).
    let buckets_size = work_units * max_buckets_per_work_unit * proj_size;
    // The amount of memory (in bytes) we need for the results.
    let results_size = work_units * proj_size;

    (max_memory - buckets_size - results_size) / term_size
}

/// The size of the exponent in bytes.
///
/// It's the actual bytes size it needs in memory, not it's theoratical bit size.
fn exp_size<F: ark_ff::PrimeField>() -> usize {
    std::mem::size_of::<F::BigInt>()
}
impl<'a, G> SingleMultiexpKernel<'a, G>
where
    //G: AffineRepr + GpuName,
    G: AffineRepr,
{
    /// Create a new Multiexp kernel instance for a device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        program: Program,
        device: &Device,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let mem = device.memory();
        let compute_units = device.compute_units();
        let compute_capability = device.compute_capability();
        let work_units = work_units(compute_units, compute_capability);
        let chunk_size = calc_chunk_size::<G>(mem, work_units);

        Ok(SingleMultiexpKernel {
            program,
            n: chunk_size,
            work_units,
            maybe_abort,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Run the actual multiexp computation on the GPU.
    ///
    /// The number of `bases` and `exponents` are determined by [`SingleMultiexpKernel`]`::n`, this
    /// means that it is guaranteed that this amount of calculations fit on the GPU this kernel is
    /// running on.
    pub fn multiexp(
        &self,
        bases: &[G],
        //exponents: &[<G::ScalarField as ark_ff::PrimeField>::BigInt],
        //exponents: &[G::ScalarField],
        exponents: &[<G::ScalarField as ark_ff::PrimeField>::BigInt],
    ) -> EcResult<G::Group> {
        assert_eq!(bases.len(), exponents.len());

        if let Some(maybe_abort) = &self.maybe_abort {
            if maybe_abort() {
                return Err(EcError::Aborted);
            }
        }
        let window_size = self.calc_window_size(bases.len());
        // windows_size * num_windows needs to be >= 256 in order for the kernel to work correctly.
        let num_windows = div_ceil(256, window_size);
        let num_groups = self.work_units / num_windows;
        let bucket_len = 1 << window_size;

        // The arkworks affine curve points have and additional struct field, this needs to be
        // stripped away in order to have a continuous stream of field elements, which is needed
        // for the GPU code.
        const ARK_G1_SIZE: usize = std::mem::size_of::<<Bls12_381 as PairingArkworks>::G1Affine>();
        const ZCASH_G1_SIZE: usize = 96;
        let bases_slice = unsafe {
            std::slice::from_raw_parts(
                bases.as_ptr() as *const _ as *const u8,
                ARK_G1_SIZE * bases.len(),
            )
        };
        let mut bases_raw = Vec::new();
        for offset in (0..bases_slice.len()).step_by(ARK_G1_SIZE) {
            bases_raw.extend_from_slice(&bases_slice[offset..(offset + ZCASH_G1_SIZE)]);
        }
        //let ark_bytes = unsafe {
        //    std::slice::from_raw_parts(ark_bases_raw.as_ptr() as *const _ as *const u8, bytes_len)
        //};

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Group>> {
            let base_buffer = program.create_buffer_from_slice(&bases_raw)?;
            let exp_buffer = program.create_buffer_from_slice(exponents)?;

            // It is safe as the GPU will initialize that buffer
            let bucket_buffer =
                unsafe { program.create_buffer::<G::Group>(self.work_units * bucket_len)? };
            // It is safe as the GPU will initialize that buffer
            let result_buffer = unsafe { program.create_buffer::<G::Group>(self.work_units)? };

            // The global work size follows CUDA's definition and is the number of
            // `LOCAL_WORK_SIZE` sized thread groups.
            let global_work_size = div_ceil(num_windows * num_groups, LOCAL_WORK_SIZE);

            //let kernel_name = format!("{}_multiexp", G::name());
            let kernel_name = "blstrs__g1__G1Affine_multiexp";
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&base_buffer)
                .arg(&bucket_buffer)
                .arg(&result_buffer)
                .arg(&exp_buffer)
                .arg(&(bases.len() as u32))
                .arg(&(num_groups as u32))
                .arg(&(num_windows as u32))
                .arg(&(window_size as u32))
                .run()?;

            let mut results = vec![G::Group::zero(); self.work_units];
            program.read_into_buffer(&result_buffer, &mut results)?;

            Ok(results)
        });

        //log::debug!("vmx: run multiexp: arkwork: gpu: start");
        let results = self.program.run(closures, ())?;
        //log::debug!("vmx: run multiexp: arkworks: gpu: stop");
        //debug!("vmx: multiexp: kernel code: result: {:?}", results);

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        // TODO vmx 2023-01-24: double check that arkworks `G::Group::zero()` corresponds to
        // group `G::Curve::identity()`
        let mut acc = G::Group::zero();
        let mut bits = 0;
        let exp_bits = exp_size::<G::ScalarField>() * 8;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc = acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }

    /// Calculates the window size, based on the given number of terms.
    ///
    /// For best performance, the window size is reduced, so that maximum parallelism is possible.
    /// If you e.g. have put only a subset of the terms into the GPU memory, then a smaller window
    /// size leads to more windows, hence more units to work on, as we split the work into
    /// `num_windows * num_groups`.
    fn calc_window_size(&self, num_terms: usize) -> usize {
        // The window size was determined by running the `gpu_multiexp_consistency` test and
        // looking at the resulting numbers.
        let window_size = ((div_ceil(num_terms, self.work_units) as f64).log2() as usize) + 2;
        std::cmp::min(window_size, MAX_WINDOW_SIZE)
    }
}

#[test]
fn gpu_multiexp_arkworks() {
    fil_logger::maybe_init();

    debug!("vmx: multiexp arkworks");

    const MAX_LOG_D: usize = 16;
    const START_LOG_D: usize = 10;

    let mut rng = rand::thread_rng();

    let mut bases = (0..(1 << START_LOG_D))
        .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
        .collect::<Vec<_>>();

    let mut ark_bases = bases
        .iter()
        .map(|base| {
            //debug!("vmx: base: {:?}", base);
            //debug!("vmx: base: {:?}", base.to_uncompressed());
            //<Bls12_381 as PairingArkworks>::G1Affine::from_random_bytes(&base.to_compressed())
            //    .unwrap()
            let x = base.x().to_bytes_le();
            let y = base.y().to_bytes_le();
            //debug!("vmx: base: x: {:?}", x);
            //debug!("vmx: base: y: {:?}", y);
            //<Bls12_381 as PairingArkworks>::G1Affine::new(<Bls12_381 as PairingArkworks>::G1Affine::BaseField::from_random_bytes(x), <Bls12_381 as PairingArkworks>::G1Affine::BaseField::from_random_bytes(y))
            //<Bls12_381 as PairingArkworks>::G1Affine::new(Fq::from_random_bytes(&x).unwrap(), Fq::from_random_bytes(&y).unwrap())
            let g1 = <Bls12_381 as PairingArkworks>::G1Affine::new(
                <Bls12_381 as PairingArkworks>::BaseField::from_random_bytes(&x).unwrap(),
                <Bls12_381 as PairingArkworks>::BaseField::from_random_bytes(&y).unwrap(),
            );
            //debug!("vmx: convertin to ark bases: arkbase: {:?}", g1.x().unwrap().into_bigint().to_bytes_le());
            //debug!("vmx: convertin to ark bases: arkbase: y: {:?}", g1.y().unwrap().into_bigint().to_bytes_le());

            //let mut g1bytes = Vec::new();
            //g1.serialize_uncompressed(&mut g1bytes);
            //debug!("vmx: g1 from x and y: {:?}", g1bytes);
            g1
        })
        .collect::<Vec<_>>();

    let mut ark_bases_cpu = bases
        .iter()
        .map(|base| {
            let x = base.x().to_bytes_le();
            let y = base.y().to_bytes_le();
            let g1 = ark_bls12_381::g1::G1Affine::new(
                <G1Config as CurveConfig>::BaseField::from_random_bytes(&x).unwrap(),
                <G1Config as CurveConfig>::BaseField::from_random_bytes(&y).unwrap(),
            );
            g1
        })
        .collect::<Vec<_>>();

    //// Find out what the underlying bytes look like
    //let bytes_len = 1920;
    //const ARK_G1_SIZE: usize = std::mem::size_of::<<Bls12_381 as PairingArkworks>::G1Affine>();
    //const ZCASH_G1_SIZE: usize = 96;
    //let ark_bases_slice = unsafe {
    //    std::slice::from_raw_parts(ark_bases.as_ptr() as *const _ as *const u8, ARK_G1_SIZE * ark_bases.len())
    //};
    //let mut ark_bases_raw = Vec::new();
    ////for i in 0..ark_bases.len() {
    //for offset in (0..ark_bases_slice.len()).step_by(ARK_G1_SIZE) {
    //    //let offset = i * ARK_G1_SIZE;
    //    //base.serialize_uncompressed(&mut ark_bases_uncompressed_vec);
    //    //debug!("vmx: ark bases: x: {:?}", base.x().unwrap().into_bigint().to_bytes_le());
    //    //debug!("vmx: ark bases: y: {:?}", base.y().unwrap().into_bigint().to_bytes_le());
    //    //debug!("vmx: ark bases: x: {:?}", base.x().unwrap().0.to_bytes_le());
    //    //debug!("vmx: ark bases: y: {:?}", base.y().unwrap().0.to_bytes_le());
    //    ark_bases_raw.extend_from_slice(&ark_bases_slice[offset..(offset + ZCASH_G1_SIZE)]);
    //}
    //let ark_bytes = unsafe {
    //    //println!("vmx: one ark base element mem size: {}", std::mem::size_of_val(&ark_bases[0]));
    //    //std::slice::from_raw_parts(ark_bases_uncompressed_vec.as_ptr() as *const _ as *const u8, bytes_len)
    //    std::slice::from_raw_parts(ark_bases_raw.as_ptr() as *const _ as *const u8, bytes_len)
    //};
    //println!("vmx: ark_bytes: {:?}", ark_bytes);
    //let zcash_bytes = unsafe {
    //    std::slice::from_raw_parts(bases.as_ptr() as *const _ as *const u8, bytes_len)
    //};
    //println!("vmx: zcash_bytes: {:?}", zcash_bytes);

    let mut kern = {
        let devices = Device::all();
        //let programs = devices
        //    .iter()
        //    .map(|device| crate::program!(device))
        //    .collect::<Result<_, _>>()
        //    .expect("Cannot create programs!");
        //MultiexpKernel::<<Bls12 as Engine>::G1Affine>::create(programs, &devices)
        //   .expect("Cannot initialize kernel!")
        let device = devices[0];
        let program = crate::program!(device).expect("Cannot create program!");
        SingleMultiexpKernelZcash::create(program, &device, None)
            .expect("Cannot initialize kernel!")
    };
    let mut ark_kern = {
        let devices = Device::all();
        let device = devices[0];
        let program = crate::program!(device).expect("Cannot create program!");
        SingleMultiexpKernel::<<Bls12_381 as PairingArkworks>::G1Affine>::create(
            program, &device, None,
        )
        .expect("Cannot initialize kernel!")
    };
    let pool = Worker::new();

    for log_d in START_LOG_D..=MAX_LOG_D {
        let samples = 1 << log_d;
        println!("Testing Multiexp for {} elements...", samples);
        let g = Arc::new(bases.clone());

        let v = Arc::new(
            (0..samples)
                .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
                .collect::<Vec<_>>(),
        );
        let ark_v = v
            .iter()
            .map(|value| {
                <Bls12_381 as PairingArkworks>::ScalarField::from_random_bytes(&value[..])
                    .unwrap()
                    .into()
            })
            .collect::<Vec<_>>();
        let ark_v_fp = ark_v
            .iter()
            .map(|value| ark_ff::Fp::from_bigint(*value).unwrap())
            .collect::<Vec<_>>();

        // Zcash
        let (zcash_result, zcash_dur) = {
            let now = Instant::now();
            let gpu = kern.multiexp(&g, &v).unwrap();
            let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("Zcash GPU took {}ms.", gpu_dur);
            (gpu, gpu_dur)
        };

        // Arkworks
        let (ark_result, ark_dur) = {
            let now = Instant::now();
            //let gpu = ark_kern.multiexp(&ark_bases[..], &ark_v[..]).unwrap();
            let gpu = G1ConfigGpu::msm(&ark_bases[..], &ark_v_fp[..]).unwrap();
            let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("Arkworks GPU took {}ms.", gpu_dur);
            (gpu, gpu_dur)
        };

        // Zcash
        let (zcash_cpu, zcash_cpu_dur) = {
            let now = Instant::now();
            let cpu = multiexp_cpu(&pool, (g.clone(), 0), FullDensity, v.clone())
                .wait()
                .unwrap();
            let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("Zcash CPU took {}ms.", cpu_dur);
            (cpu, cpu_dur)
        };

        // Arkworks
        let (ark_cpu, ark_cpu_dur) = {
            let now = Instant::now();
            let cpu = G1Config::msm(&ark_bases_cpu[..], &ark_v_fp[..]).unwrap();
            let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("Arkworks CPU took {}ms.", cpu_dur);
            (cpu, cpu_dur)
        };

        println!(
            "Arkworks Speedup CPU to GPU: x{}",
            ark_cpu_dur as f32 / ark_dur as f32
        );
        println!(
            "Zcash Speedup CPU to GPU: x{}",
            zcash_cpu_dur as f32 / zcash_dur as f32
        );
        println!(
            "Arkworks Speedup GPU to Zcash GPU: x{}",
            ark_dur as f32 / zcash_dur as f32
        );

        //println!("vmx: ark_result: {:?}", ark_result.into_affine());
        let zcash_result_as_arkworks = {
            let affine = zcash_result.to_affine();
            let x = affine.x().to_bytes_le();
            let y = affine.y().to_bytes_le();
            <Bls12_381 as PairingArkworks>::G1Affine::new(
                <Bls12_381 as PairingArkworks>::BaseField::from_random_bytes(&x).unwrap(),
                <Bls12_381 as PairingArkworks>::BaseField::from_random_bytes(&y).unwrap(),
            )
        };
        let zcash_result_as_arkworks_cpu = {
            let affine = zcash_result.to_affine();
            let x = affine.x().to_bytes_le();
            let y = affine.y().to_bytes_le();
            ark_bls12_381::g1::G1Affine::new(
                <G1Config as CurveConfig>::BaseField::from_random_bytes(&x).unwrap(),
                <G1Config as CurveConfig>::BaseField::from_random_bytes(&y).unwrap(),
            )
        };

        assert_eq!(zcash_result_as_arkworks, ark_result.into_affine());
        assert_eq!(zcash_result_as_arkworks_cpu, ark_cpu.into_affine());
        assert_eq!(zcash_cpu, zcash_result);

        println!("============================");

        bases = [bases.clone(), bases.clone()].concat();
        ark_bases = [ark_bases.clone(), ark_bases.clone()].concat();
        ark_bases_cpu = [ark_bases_cpu.clone(), ark_bases_cpu.clone()].concat();
    }
}
