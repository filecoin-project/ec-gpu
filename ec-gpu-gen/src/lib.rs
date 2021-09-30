mod nvidia;

use ec_gpu::{GpuEngine, GpuField};

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");
static FIELD2_SRC: &str = include_str!("cl/field2.cl");
static EC_SRC: &str = include_str!("cl/ec.cl");

/// Generates the source for the elliptic curve and group operations, as defined by `E`.
///
/// The code from the [`common()`] call needs to be included before this on is used.
pub fn gen_ec_source<E: GpuEngine, L: Limb>() -> String {
    vec![
        field::<E::Scalar, L>("Fr"),
        field::<E::Fp, L>("Fq"),
        field2("Fq2", "Fq"),
        ec("Fq", "G1"),
        ec("Fq2", "G2"),
    ]
    .join("\n\n")
}

fn ec(field: &str, point: &str) -> String {
    String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point)
}

fn field2(field2: &str, field: &str) -> String {
    String::from(FIELD2_SRC)
        .replace("FIELD2", field2)
        .replace("FIELD", field)
}

pub trait Limb: Sized + Clone + Copy {
    type LimbType: Clone + std::fmt::Display;
    fn zero() -> Self;
    fn new(val: Self::LimbType) -> Self;
    fn value(&self) -> Self::LimbType;
    fn bits() -> usize;
    fn ptx_info() -> (&'static str, &'static str);
    fn opencl_type() -> &'static str;
    fn one_limbs<F: GpuField>() -> Vec<Self>;
    // Returns the field modulus in non-Montgomery form as a vector of `Self::LimbType` (least
    // significant limb first).
    fn modulus_limbs<F: GpuField>() -> Vec<Self>;
    /// Calculate the `INV` parameter of Montgomery reduction algorithm for 32/64bit limbs
    /// * `a` - Is the first limb of modulus
    fn calc_inv(a: Self) -> Self;
    fn calculate_r2<F: GpuField>() -> Vec<Self>;
}

#[derive(Clone, Copy)]
pub struct Limb32(u32);
impl Limb for Limb32 {
    type LimbType = u32;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn bits() -> usize {
        32
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u32", "r")
    }
    fn opencl_type() -> &'static str {
        "uint"
    }
    fn one_limbs<F: GpuField>() -> Vec<Self> {
        F::one().into_iter().map(Self::new).collect()
    }
    fn modulus_limbs<F: GpuField>() -> Vec<Self> {
        F::modulus().into_iter().map(Self::new).collect()
    }
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u32;
        for _ in 0..31 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn calculate_r2<F: GpuField>() -> Vec<Self> {
        F::r2().into_iter().map(Self::new).collect()
    }
}

#[derive(Clone, Copy)]
pub struct Limb64(u64);
impl Limb for Limb64 {
    type LimbType = u64;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn bits() -> usize {
        64
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u64", "l")
    }
    fn opencl_type() -> &'static str {
        "ulong"
    }
    fn one_limbs<F: GpuField>() -> Vec<Self> {
        F::one()
            .chunks(2)
            .map(|chunk| Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64)))
            .collect()
    }

    fn modulus_limbs<F: GpuField>() -> Vec<Self> {
        F::modulus()
            .chunks(2)
            .map(|chunk| Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64)))
            .collect()
    }

    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn calculate_r2<F: GpuField>() -> Vec<Self> {
        F::r2()
            .chunks(2)
            .map(|chunk| Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64)))
            .collect()
    }
}

fn define_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "#define {} ((FIELD){{ {{ {} }} }})",
        name,
        limbs
            .iter()
            .map(|l| l.value().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Generates CUDA/OpenCL constants and type definitions of prime-field `F`
fn params<F, L: Limb>() -> String
where
    F: GpuField,
{
    let one = L::one_limbs::<F>(); // Get Montgomery form of F::one()
    let p = L::modulus_limbs::<F>(); // Get field modulus in non-Montgomery form
    let r2 = L::calculate_r2::<F>();
    let limbs = one.len(); // Number of limbs
    let inv = L::calc_inv(p[0]);
    let limb_def = format!("#define FIELD_limb {}", L::opencl_type());
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let limb_bits_def = format!("#define FIELD_LIMB_BITS {}", L::bits());
    let p_def = define_field("FIELD_P", p);
    let r2_def = define_field("FIELD_R2", r2);
    let one_def = define_field("FIELD_ONE", one);
    let zero_def = define_field("FIELD_ZERO", vec![L::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv.value());
    let typedef = "typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;".to_string();
    [
        limb_def,
        limbs_def,
        limb_bits_def,
        one_def,
        p_def,
        r2_def,
        zero_def,
        inv_def,
        typedef,
    ]
    .join("\n")
}

/// Returns CUDA/OpenCL source-code of a ff::PrimeField with name `name`
/// Find details in README.md
///
/// The code from the [`common()`] call needs to be included before this on is used.
pub fn field<F, L: Limb>(name: &str) -> String
where
    F: GpuField,
{
    [
        params::<F, L>(),
        nvidia::field_add_sub_nvidia::<F, L>().expect("preallocated"),
        String::from(FIELD_SRC),
    ]
    .join("\n")
    .replace("FIELD", name)
}

/// Returns CUDA/OpenCL source-code that contains definitions/functions that are shared across
/// fields.
///
/// It needs to be called before any other function like [`field`] or [`gen_ec_source`] is called,
/// as it contains deinitions, used in those.
pub fn common() -> String {
    COMMON_SRC.to_string()
}

#[cfg(all(test, any(feature = "tests-opencl", feature = "tests-cuda")))]
mod tests {
    use super::*;

    use std::sync::Mutex;

    #[cfg(feature = "tests-cuda")]
    use rust_gpu_tools::cuda;
    #[cfg(feature = "tests-opencl")]
    use rust_gpu_tools::opencl;
    use rust_gpu_tools::{program_closures, Device, GPUError, Program};

    use blstrs::Scalar;
    use ff::{Field, PrimeField};
    use lazy_static::lazy_static;
    use rand::{thread_rng, Rng};

    static TEST_SRC: &str = include_str!("./cl/test.cl");

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(transparent)]
    pub struct GpuScalar(pub Scalar);
    impl Default for GpuScalar {
        fn default() -> Self {
            Self(Scalar::zero())
        }
    }

    #[cfg(feature = "tests-cuda")]
    impl cuda::KernelArgument for GpuScalar {
        fn as_c_void(&self) -> *mut std::ffi::c_void {
            &self.0 as *const _ as _
        }
    }

    #[cfg(feature = "tests-opencl")]
    impl opencl::KernelArgument for GpuScalar {
        fn push(&self, kernel: &mut opencl::Kernel) {
            kernel.builder.set_arg(&self.0);
        }
    }

    /// The `run` call needs to return a result, use this struct as placeholder.
    #[derive(Debug)]
    struct NoError;
    impl From<GPUError> for NoError {
        fn from(_error: GPUError) -> Self {
            Self
        }
    }

    // CUDA doesn't support 64-bit limbs
    #[cfg(feature = "tests-cuda")]
    fn source_cuda() -> String {
        let src = vec![
            common(),
            field::<Scalar, Limb32>("Scalar32"),
            TEST_SRC.to_string(),
        ]
        .join("\n\n");
        println!("{}", src);
        src
    }

    #[cfg(feature = "tests-opencl")]
    fn source_opencl() -> String {
        let src = vec![
            common(),
            field::<Scalar, Limb32>("Scalar32"),
            field::<Scalar, Limb64>("Scalar64"),
            TEST_SRC.to_string(),
        ]
        .join("\n\n");
        println!("{}", src);
        src
    }

    #[cfg(feature = "tests-cuda")]
    lazy_static! {
        static ref CUDA_PROGRAM: Mutex<Program> = {
            use std::ffi::CString;
            use std::fs;
            use std::process::Command;

            let tmpdir = tempfile::tempdir().expect("Cannot create temporary directory.");
            let source_path = tmpdir.path().join("kernel.cu");
            fs::write(&source_path, source_cuda().as_bytes())
                .expect("Cannot write kernel source file.");
            let fatbin_path = tmpdir.path().join("kernel.fatbin");

            let nvcc = Command::new("nvcc")
                .arg("--fatbin")
                .arg("--gpu-architecture=sm_86")
                .arg("--generate-code=arch=compute_86,code=sm_86")
                .arg("--generate-code=arch=compute_80,code=sm_80")
                .arg("--generate-code=arch=compute_75,code=sm_75")
                .arg("--output-file")
                .arg(&fatbin_path)
                .arg(&source_path)
                .status()
                .expect("Cannot run nvcc.");

            if !nvcc.success() {
                panic!(
                    "nvcc failed. See the kernel source at {}.",
                    source_path.display()
                );
            }

            let device = *Device::all().first().expect("Cannot get a default device.");
            let cuda_device = device.cuda_device().unwrap();
            let fatbin_path_cstring =
                CString::new(fatbin_path.to_str().expect("path is not valid UTF-8."))
                    .expect("path contains NULL byte.");
            let program =
                cuda::Program::from_binary(cuda_device, fatbin_path_cstring.as_c_str()).unwrap();
            Mutex::new(Program::Cuda(program))
        };
    }

    #[cfg(feature = "tests-opencl")]
    lazy_static! {
        static ref OPENCL_PROGRAM: Mutex<Program> = {
            let device = *Device::all().first().expect("Cannot get a default device");
            let opencl_device = device.opencl_device().unwrap();
            let program = opencl::Program::from_opencl(opencl_device, &source_opencl()).unwrap();
            Mutex::new(Program::Opencl(program))
        };
    }

    fn call_kernel(name: &str, scalars: &[GpuScalar], uints: &[u32]) -> Scalar {
        let closures = program_closures!(|program, _args| -> Result<Scalar, NoError> {
            let mut cpu_buffer = vec![GpuScalar::default()];
            let buffer = program.create_buffer_from_slice(&cpu_buffer).unwrap();

            let mut kernel = program.create_kernel(name, 1, 64).unwrap();
            for scalar in scalars {
                kernel = kernel.arg(scalar);
            }
            for uint in uints {
                kernel = kernel.arg(uint);
            }
            kernel.arg(&buffer).run().unwrap();

            program.read_into_buffer(&buffer, &mut cpu_buffer).unwrap();
            Ok(cpu_buffer[0].0)
        });

        #[cfg(all(feature = "tests-cuda", not(feature = "tests-opencl")))]
        return CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();

        #[cfg(all(feature = "tests-opencl", not(feature = "tests-cuda")))]
        return OPENCL_PROGRAM.lock().unwrap().run(closures, ()).unwrap();

        // When both features are enabled, check if the results are the same
        #[cfg(all(feature = "tests-cuda", feature = "tests-opencl"))]
        {
            let cuda_result = CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();
            let opencl_result = OPENCL_PROGRAM.lock().unwrap().run(closures, ()).unwrap();
            assert_eq!(cuda_result, opencl_result);
            cuda_result
        }
    }

    #[test]
    fn test_add() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = Scalar::random(&mut rng);
            let c = a + b;

            assert_eq!(
                call_kernel("test_add_32", &vec![GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(
                call_kernel("test_add_64", &vec![GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = Scalar::random(&mut rng);
            let c = a - b;
            assert_eq!(
                call_kernel("test_sub_32", &vec![GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(
                call_kernel("test_sub_64", &vec![GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = Scalar::random(&mut rng);
            let c = a * b;

            assert_eq!(
                call_kernel("test_mul_32", &vec![GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(
                call_kernel("test_mul_64", &vec![GpuScalar(a), GpuScalar(b)], &[]),
                c
            );
        }
    }

    #[test]
    fn test_pow() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = rng.gen::<u32>();
            let c = a.pow_vartime([b as u64]);
            assert_eq!(call_kernel("test_pow_32", &vec![GpuScalar(a)], &vec![b]), c);
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(call_kernel("test_pow_64", &vec![GpuScalar(a)], &vec![b]), c);
        }
    }

    #[test]
    fn test_sqr() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = a.square();

            assert_eq!(call_kernel("test_sqr_32", &vec![GpuScalar(a)], &[]), b);
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(call_kernel("test_sqr_64", &vec![GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = a.double();

            assert_eq!(call_kernel("test_double_32", &vec![GpuScalar(a)], &[]), b);
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(call_kernel("test_double_64", &vec![GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_unmont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b: Scalar = unsafe { std::mem::transmute(a.to_repr()) };
            assert_eq!(call_kernel("test_unmont_32", &vec![GpuScalar(a)], &[]), b);
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(call_kernel("test_unmont_64", &vec![GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_mont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a_repr = Scalar::random(&mut rng).to_repr();
            let a: Scalar = unsafe { std::mem::transmute(a_repr) };
            let b = Scalar::from_repr(a_repr).unwrap();
            assert_eq!(call_kernel("test_mont_32", &vec![GpuScalar(a)], &[]), b);
            #[cfg(not(feature = "tests-cuda"))]
            assert_eq!(call_kernel("test_mont_64", &vec![GpuScalar(a)], &[]), b);
        }
    }
}
