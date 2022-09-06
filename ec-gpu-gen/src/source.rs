use std::collections::HashSet;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
#[cfg(any(feature = "opencl", feature = "cuda"))]
use std::path::PathBuf;
#[cfg(any(feature = "opencl", feature = "cuda"))]
use std::{env, fs};

use ec_gpu::{GpuField, GpuName};
use group::prime::PrimeCurveAffine;

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");
static FIELD2_SRC: &str = include_str!("cl/field2.cl");
static EC_SRC: &str = include_str!("cl/ec.cl");
static FFT_SRC: &str = include_str!("cl/fft.cl");
static MULTIEXP_SRC: &str = include_str!("cl/multiexp.cl");

#[derive(Clone, Copy)]
enum Limb32Or64 {
    Limb32,
    Limb64,
}

/// This trait is used to uniquely identify items by some identifier (`name`) and to return the GPU
/// source code they produce.
trait NameAndSource {
    /// The name to identify the item.
    fn name(&self) -> String;
    /// The GPU source code that is generated.
    fn source(&self, limb: Limb32Or64) -> String;
}

impl PartialEq for dyn NameAndSource {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for dyn NameAndSource {}

impl Hash for dyn NameAndSource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

/// Prints the name by default, the source code of the 32-bit limb in the alternate mode via
/// `{:#?}`.
impl fmt::Debug for dyn NameAndSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.debug_map()
                .entries(vec![
                    ("name", self.name()),
                    ("source", self.source(Limb32Or64::Limb32)),
                ])
                .finish()
        } else {
            write!(f, "{:?}", self.name())
        }
    }
}

/// A field that might also be an extension field.
///
/// When the field is an extension field, we also add its sub-field to the list of fields. This
/// enum is used to indicate that it's a sub-field that has a corresponding extension field. This
/// way we can make sure that when the source is generated, that also the source for the sub-field
/// is generated, while not having duplicated field definitions.
// Storing the sub-field as a string is a bit of a hack around Rust's type system. If we would
// store the generic type, then the enum would need to be generic over two fields, even in
// the case when no extension field is used. This would make the API harder to use.
#[derive(Debug)]
enum Field<F: GpuField> {
    /// A field, might be an extension field.
    Field(PhantomData<F>),
    /// A sub-field with the given name that has a corresponding extension field.
    SubField(String),
}

impl<F: GpuField> Field<F> {
    /// Create a new field for the given generic type.
    pub fn new() -> Self {
        // By default it's added as a field. If it's an extension field, then the `add_field()`
        // function will create a copy of it, as `SubField` variant.
        Self::Field(PhantomData)
    }
}

impl<F: GpuField> Default for Field<F> {
    fn default() -> Self {
        Self::new()
    }
}

fn field_source<F: GpuField>(limb: Limb32Or64) -> String {
    match limb {
        Limb32Or64::Limb32 => [
            params::<F, Limb32>(),
            field_add_sub_nvidia::<F, Limb32>().expect("preallocated"),
            String::from(FIELD_SRC),
        ]
        .join("\n"),
        Limb32Or64::Limb64 => [
            params::<F, Limb64>(),
            field_add_sub_nvidia::<F, Limb64>().expect("preallocated"),
            String::from(FIELD_SRC),
        ]
        .join("\n"),
    }
}

impl<F: GpuField> NameAndSource for Field<F> {
    fn name(&self) -> String {
        match self {
            Self::Field(_) => F::name(),
            Self::SubField(name) => name.to_string(),
        }
    }

    fn source(&self, limb: Limb32Or64) -> String {
        match self {
            Self::Field(_) => {
                // If it's an extension field.
                if let Some(sub_field_name) = F::sub_field_name() {
                    String::from(FIELD2_SRC)
                        .replace("FIELD2", &F::name())
                        .replace("FIELD", &sub_field_name)
                } else {
                    field_source::<F>(limb).replace("FIELD", &F::name())
                }
            }
            Self::SubField(sub_field_name) => {
                // The `GpuField` implementation of the extension field contains the constants of
                // the sub-field. Hence we can just forward the `F`. It's important that those
                // functions do *not* use the name of the field, else we might generate the
                // sub-field named like the extension field.
                field_source::<F>(limb).replace("FIELD", sub_field_name)
            }
        }
    }
}

/// Struct that generates FFT GPU source code.
struct Fft<F: GpuName>(PhantomData<F>);

impl<F: GpuName> NameAndSource for Fft<F> {
    fn name(&self) -> String {
        F::name()
    }

    fn source(&self, _limb: Limb32Or64) -> String {
        String::from(FFT_SRC).replace("FIELD", &F::name())
    }
}

/// Struct that generates multiexp GPU smource code.
struct Multiexp<P: GpuName, F: GpuName, Exp: GpuName> {
    curve_point: PhantomData<P>,
    field: PhantomData<F>,
    exponent: PhantomData<Exp>,
}

impl<P: GpuName, F: GpuName, Exp: GpuName> Multiexp<P, F, Exp> {
    pub fn new() -> Self {
        Self {
            curve_point: PhantomData::<P>,
            field: PhantomData::<F>,
            exponent: PhantomData::<Exp>,
        }
    }
}

impl<P: GpuName, F: GpuName, Exp: GpuName> NameAndSource for Multiexp<P, F, Exp> {
    fn name(&self) -> String {
        P::name()
    }

    fn source(&self, _limb: Limb32Or64) -> String {
        let ec = String::from(EC_SRC)
            .replace("FIELD", &F::name())
            .replace("POINT", &P::name());
        let multiexp = String::from(MULTIEXP_SRC)
            .replace("POINT", &P::name())
            .replace("EXPONENT", &Exp::name());
        [ec, multiexp].concat()
    }
}

/// Builder to create the source code of a GPU kernel.
///
/// # Example
///
/// ```
/// use blstrs::{Fp, Fp2, G1Affine, G2Affine, Scalar};
/// use ec_gpu_gen::SourceBuilder;
///
/// # #[cfg(any(feature = "cuda", feature = "opencl"))]
/// let source = SourceBuilder::new()
///     .add_fft::<Scalar>()
///     .add_multiexp::<G1Affine, Fp>()
///     .add_multiexp::<G2Affine, Fp2>()
///     .build_32_bit_limbs();
///```
// In the `HashSet`s the concrete types cannot be used, as each item of the set should be able to
// have its own (different) generic type.
// We distinguish between extension fields and other fields as sub-fields need to be defined first
// in the source code (due to being C, where the order of declaration matters).
pub struct SourceBuilder {
    /// The [`Field`]s that are used in this kernel.
    fields: HashSet<Box<dyn NameAndSource>>,
    /// The extension [`Field`]s that are used in this kernel.
    extension_fields: HashSet<Box<dyn NameAndSource>>,
    /// The [`Fft`]s that are used in this kernel.
    ffts: HashSet<Box<dyn NameAndSource>>,
    /// The [`Multiexp`]s that are used in this kernel.
    multiexps: HashSet<Box<dyn NameAndSource>>,
    /// Additional source that is appended at the end of the generated source.
    extra_sources: Vec<String>,
}

impl SourceBuilder {
    /// Create a new configuration to generation a GPU kernel.
    pub fn new() -> Self {
        Self {
            fields: HashSet::new(),
            extension_fields: HashSet::new(),
            ffts: HashSet::new(),
            multiexps: HashSet::new(),
            extra_sources: Vec::new(),
        }
    }

    /// Add a field to the configuration.
    ///
    /// If it is an extension field, then the extension field *and* the sub-field is added.
    pub fn add_field<F>(mut self) -> Self
    where
        F: GpuField + 'static,
    {
        let field = Field::<F>::new();
        // If it's an extension field, also add the corresponding sub-field.
        if let Some(sub_field_name) = F::sub_field_name() {
            self.extension_fields.insert(Box::new(field));
            let sub_field = Field::<F>::SubField(sub_field_name);
            self.fields.insert(Box::new(sub_field));
        } else {
            self.fields.insert(Box::new(field));
        }
        self
    }

    /// Add an FFT kernel function to the configuration.
    pub fn add_fft<F>(self) -> Self
    where
        F: GpuField + 'static,
    {
        let mut config = self.add_field::<F>();
        let fft = Fft::<F>(PhantomData);
        config.ffts.insert(Box::new(fft));
        config
    }

    /// Add an Multiexp kernel function to the configuration.
    ///
    /// The field must be given explicitly as currently it cannot derived from the curve point
    /// directly.
    pub fn add_multiexp<C, F>(self) -> Self
    where
        C: PrimeCurveAffine + GpuName,
        C::Scalar: GpuField,
        F: GpuField + 'static,
    {
        let mut config = self.add_field::<F>().add_field::<C::Scalar>();
        let multiexp = Multiexp::<C, F, C::Scalar>::new();
        config.multiexps.insert(Box::new(multiexp));
        config
    }

    /// Appends some given source at the end of the generated source.
    ///
    /// This is useful for cases where you use this library as building block, but have your own
    /// kernel implementation. If this function is is called several times, then those sources are
    /// appended in that call order.
    pub fn append_source(mut self, source: String) -> Self {
        self.extra_sources.push(source);
        self
    }

    /// Generate the GPU kernel source code based on the current configuration with 32-bit limbs.
    ///
    /// On CUDA 32-bit limbs are recommended.
    pub fn build_32_bit_limbs(&self) -> String {
        self.build(Limb32Or64::Limb32)
    }

    /// Generate the GPU kernel source code based on the current configuration with 64-bit limbs.
    ///
    /// On OpenCL 32-bit limbs are recommended.
    pub fn build_64_bit_limbs(&self) -> String {
        self.build(Limb32Or64::Limb64)
    }

    /// Generate the GPU kernel source code based on the current configuration.
    fn build(&self, limb_size: Limb32Or64) -> String {
        let fields = self
            .fields
            .iter()
            .map(|field| field.source(limb_size))
            .collect();
        let extension_fields = self
            .extension_fields
            .iter()
            .map(|field| field.source(limb_size))
            .collect();
        let ffts = self.ffts.iter().map(|fft| fft.source(limb_size)).collect();
        let multiexps = self
            .multiexps
            .iter()
            .map(|multiexp| multiexp.source(limb_size))
            .collect();
        let extra_sources = self.extra_sources.join("\n");
        vec![
            COMMON_SRC.to_string(),
            fields,
            extension_fields,
            ffts,
            multiexps,
            extra_sources,
        ]
        .join("\n\n")
    }
}

impl Default for SourceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait to implement limbs of different underlying bit sizes.
pub trait Limb: Sized + Clone + Copy {
    /// The underlying size of the limb, e.g. `u32`
    type LimbType: Clone + std::fmt::Display;
    /// Returns the value representing zero.
    fn zero() -> Self;
    /// Returns a new limb.
    fn new(val: Self::LimbType) -> Self;
    /// Returns the raw value of the limb.
    fn value(&self) -> Self::LimbType;
    /// Returns the bit size of the limb.
    fn bits() -> usize {
        mem::size_of::<Self::LimbType>() * 8
    }
    /// Returns a tuple with the strings that PTX is using to describe the type and the register.
    fn ptx_info() -> (&'static str, &'static str);
    /// Returns the type that OpenCL is using to represent the limb.
    fn opencl_type() -> &'static str;
    /// Returns the limbs that represent the multiplicative identity of the given field.
    fn one_limbs<F: GpuField>() -> Vec<Self>;
    /// Returns the field modulus in non-Montgomery form as a vector of `Self::LimbType` (least
    /// significant limb first).
    fn modulus_limbs<F: GpuField>() -> Vec<Self>;
    /// Calculate the `INV` parameter of Montgomery reduction algorithm for 32/64bit limbs
    /// * `a` - Is the first limb of modulus.
    fn calc_inv(a: Self) -> Self;
    /// Returns the limbs that represent `R ^ 2 mod P`.
    fn calculate_r2<F: GpuField>() -> Vec<Self>;
}

/// A 32-bit limb.
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

/// A 64-bit limb.
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

fn const_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "CONSTANT FIELD {} = {{ {{ {} }} }};",
        name,
        limbs
            .iter()
            .map(|l| l.value().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Generates CUDA/OpenCL constants and type definitions of prime-field `F`
fn params<F, L>() -> String
where
    F: GpuField,
    L: Limb,
{
    let one = L::one_limbs::<F>(); // Get Montgomery form of F::one()
    let p = L::modulus_limbs::<F>(); // Get field modulus in non-Montgomery form
    let r2 = L::calculate_r2::<F>();
    let limbs = one.len(); // Number of limbs
    let inv = L::calc_inv(p[0]);
    let limb_def = format!("#define FIELD_limb {}", L::opencl_type());
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let limb_bits_def = format!("#define FIELD_LIMB_BITS {}", L::bits());
    let p_def = const_field("FIELD_P", p);
    let r2_def = const_field("FIELD_R2", r2);
    let one_def = const_field("FIELD_ONE", one);
    let zero_def = const_field("FIELD_ZERO", vec![L::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv.value());
    let typedef = "typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;".to_string();
    [
        limb_def,
        limbs_def,
        limb_bits_def,
        inv_def,
        typedef,
        one_def,
        p_def,
        r2_def,
        zero_def,
    ]
    .join("\n")
}

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
fn field_add_sub_nvidia<F, L>() -> Result<String, std::fmt::Error>
where
    F: GpuField,
    L: Limb,
{
    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_info();

    writeln!(result, "#if defined(OPENCL_NVIDIA) || defined(CUDA)\n")?;
    for op in &["sub", "add"] {
        let len = L::one_limbs::<F>().len();

        writeln!(
            result,
            "DEVICE FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{",
            op
        )?;
        if len > 1 {
            write!(result, "asm(")?;
            writeln!(result, "\"{}.cc.{} %0, %0, %{};\\r\\n\"", op, ptx_type, len)?;

            for i in 1..len - 1 {
                writeln!(
                    result,
                    "\"{}c.cc.{} %{}, %{}, %{};\\r\\n\"",
                    op,
                    ptx_type,
                    i,
                    i,
                    len + i
                )?;
            }
            writeln!(
                result,
                "\"{}c.{} %{}, %{}, %{};\\r\\n\"",
                op,
                ptx_type,
                len - 1,
                len - 1,
                2 * len - 1
            )?;

            write!(result, ":")?;
            for n in 0..len {
                write!(result, "\"+{}\"(a.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }

            write!(result, "\n:")?;
            for n in 0..len {
                write!(result, "\"{}\"(b.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }
            writeln!(result, ");")?;
        }
        writeln!(result, "return a;\n}}")?;
    }
    writeln!(result, "#endif")?;

    Ok(result)
}

/// Convience function to generate a kernel/source based on a source builder.
///
/// When the `cuda` feature is enabled it will compile a CUDA fatbin. The path to the file is
/// stored in the `_EC_GPU_CUDA_KERNEL_FATBIN` environment variable, that will automatically be
/// used by the `ec-gpu-gen` functionality that needs a kernel.
///
///
/// When the `opencl` feature is enabled it will generate the source code for OpenCL. The path to
/// the source file is stored in the `_EC_GPU_OPENCL_KERNEL_SOURCE` environment variable, that will
/// automatically be used by the `ec-gpu-gen` functionality that needs a kernel. OpenCL compiles
/// the source at run time).
#[allow(unused_variables)]
pub fn generate(source_builder: &SourceBuilder) {
    #[cfg(feature = "cuda")]
    generate_cuda(source_builder);
    #[cfg(feature = "opencl")]
    generate_opencl(source_builder);
}

#[cfg(feature = "cuda")]
fn generate_cuda(source_builder: &SourceBuilder) -> PathBuf {
    use sha2::{Digest, Sha256};

    // This is a hack when no properly compiled kernel is needed. That's the case when the
    // documentation is built on docs.rs and when Clippy is run. We can use arbitrary bytes as
    // input then.
    if env::var("DOCS_RS").is_ok() || cfg!(feature = "cargo-clippy") {
        println!("cargo:rustc-env=_EC_GPU_CUDA_KERNEL_FATBIN=../build.rs");
        return PathBuf::from("../build.rs");
    }

    let kernel_source = source_builder.build_32_bit_limbs();
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR was not set.");

    // Make it possible to override the default options. Though the source and output file is
    // always set automatically.
    let mut nvcc = match env::var("EC_GPU_CUDA_NVCC_ARGS") {
        Ok(args) => execute::command(format!("nvcc {}", args)),
        Err(_) => {
            let mut command = std::process::Command::new("nvcc");
            command
                .arg("--optimize=6")
                // Compile with as many threads as CPUs are available.
                .arg("--threads=0")
                .arg("--fatbin")
                .arg("--gpu-architecture=sm_86")
                .arg("--generate-code=arch=compute_86,code=sm_86")
                .arg("--generate-code=arch=compute_80,code=sm_80")
                .arg("--generate-code=arch=compute_75,code=sm_75");
            command
        }
    };

    // Hash the source and the compile flags. Use that as the filename, so that the kernel is only
    // rebuilt if any of them change.
    let mut hasher = Sha256::new();
    hasher.update(kernel_source.as_bytes());
    hasher.update(&format!("{:?}", &nvcc));
    let kernel_digest = hex::encode(hasher.finalize());

    let source_path: PathBuf = [&out_dir, &format!("{}.cu", &kernel_digest)]
        .iter()
        .collect();
    let fatbin_path: PathBuf = [&out_dir, &format!("{}.fatbin", &kernel_digest)]
        .iter()
        .collect();

    fs::write(&source_path, &kernel_source).unwrap_or_else(|_| {
        panic!(
            "Cannot write kernel source at {}.",
            source_path.to_str().unwrap()
        )
    });

    // Only compile if the output doesn't exist yet.
    if !fatbin_path.as_path().exists() {
        let status = nvcc
            .arg("--output-file")
            .arg(&fatbin_path)
            .arg(&source_path)
            .status()
            .expect("Cannot run nvcc. Install the NVIDIA toolkit or disable the `cuda` feature.");

        if !status.success() {
            panic!(
                "nvcc failed. See the kernel source at {}",
                source_path.to_str().unwrap()
            );
        }
    }

    // The idea to put the path to the farbin into a compile-time env variable is from
    // https://github.com/LutzCle/fast-interconnects-demo/blob/b80ea8e04825167f486ab8ac1b5d67cf7dd51d2c/rust-demo/build.rs
    println!(
        "cargo:rustc-env=_EC_GPU_CUDA_KERNEL_FATBIN={}",
        fatbin_path.to_str().unwrap()
    );

    fatbin_path
}

#[cfg(feature = "opencl")]
fn generate_opencl(source_builder: &SourceBuilder) -> PathBuf {
    let kernel_source = source_builder.build_64_bit_limbs();
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR was not set.");

    // Generating the kernel source is cheap, hence use a fixed name and override it on every
    // build.
    let source_path: PathBuf = [&out_dir, "kernel.cl"].iter().collect();

    fs::write(&source_path, &kernel_source).unwrap_or_else(|_| {
        panic!(
            "Cannot write kernel source at {}.",
            source_path.to_str().unwrap()
        )
    });

    // For OpenCL we only need the kernel source, it is compiled at runtime.
    println!(
        "cargo:rustc-env=_EC_GPU_OPENCL_KERNEL_SOURCE={}",
        source_path.to_str().unwrap()
    );

    source_path
}

#[cfg(all(test, any(feature = "opencl", feature = "cuda")))]
mod tests {
    use super::*;

    use std::sync::Mutex;

    #[cfg(feature = "cuda")]
    use rust_gpu_tools::cuda;
    #[cfg(feature = "opencl")]
    use rust_gpu_tools::opencl;
    use rust_gpu_tools::{program_closures, Device, GPUError, Program};

    use blstrs::Scalar;
    use ff::{Field as _, PrimeField};
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

    #[cfg(feature = "cuda")]
    impl cuda::KernelArgument for GpuScalar {
        fn as_c_void(&self) -> *mut std::ffi::c_void {
            &self.0 as *const _ as _
        }
    }

    #[cfg(feature = "opencl")]
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

    fn test_source() -> SourceBuilder {
        let test_source = String::from(TEST_SRC).replace("FIELD", &Scalar::name());
        SourceBuilder::new()
            .add_field::<Scalar>()
            .append_source(test_source)
    }

    #[cfg(feature = "cuda")]
    lazy_static! {
        static ref CUDA_PROGRAM: Mutex<Program> = {
            use std::ffi::CString;

            let source = test_source();
            let fatbin_path = generate_cuda(&source);

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

    #[cfg(feature = "opencl")]
    lazy_static! {
        static ref OPENCL_PROGRAM: Mutex<(Program, Program)> = {
            let device = *Device::all().first().expect("Cannot get a default device");
            let opencl_device = device.opencl_device().unwrap();
            let source_32 = test_source().build_32_bit_limbs();
            let program_32 = opencl::Program::from_opencl(opencl_device, &source_32).unwrap();
            let source_64 = test_source().build_64_bit_limbs();
            let program_64 = opencl::Program::from_opencl(opencl_device, &source_64).unwrap();
            Mutex::new((Program::Opencl(program_32), Program::Opencl(program_64)))
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

        // For CUDA we only test 32-bit limbs.
        #[cfg(all(feature = "cuda", not(feature = "opencl")))]
        return CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();

        // For OpenCL we test for 32 and 64-bi limbs.
        #[cfg(all(feature = "opencl", not(feature = "cuda")))]
        {
            let result_32 = OPENCL_PROGRAM.lock().unwrap().0.run(closures, ()).unwrap();
            let result_64 = OPENCL_PROGRAM.lock().unwrap().1.run(closures, ()).unwrap();
            assert_eq!(
                result_32, result_64,
                "Results for 32-bit and 64-bit limbs must be the same."
            );
            result_32
        }

        // When both features are enabled, check if the results are the same
        #[cfg(all(feature = "cuda", feature = "opencl"))]
        {
            let cuda_result = CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();
            let opencl_32_result = OPENCL_PROGRAM.lock().unwrap().0.run(closures, ()).unwrap();
            let opencl_64_result = OPENCL_PROGRAM.lock().unwrap().1.run(closures, ()).unwrap();
            assert_eq!(
                opencl_32_result, opencl_64_result,
                "Results for 32-bit and 64-bit limbs on OpenCL must be the same."
            );
            assert_eq!(
                cuda_result, opencl_32_result,
                "Results for CUDA and OpenCL must be the same."
            );
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
                call_kernel("test_add", &[GpuScalar(a), GpuScalar(b)], &[]),
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
                call_kernel("test_sub", &[GpuScalar(a), GpuScalar(b)], &[]),
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
                call_kernel("test_mul", &[GpuScalar(a), GpuScalar(b)], &[]),
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
            assert_eq!(call_kernel("test_pow", &[GpuScalar(a)], &[b]), c);
        }
    }

    #[test]
    fn test_sqr() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = a.square();

            assert_eq!(call_kernel("test_sqr", &[GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b = a.double();

            assert_eq!(call_kernel("test_double", &[GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_unmont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Scalar::random(&mut rng);
            let b: Scalar = unsafe { std::mem::transmute(a.to_repr()) };
            assert_eq!(call_kernel("test_unmont", &[GpuScalar(a)], &[]), b);
        }
    }

    #[test]
    fn test_mont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a_repr = Scalar::random(&mut rng).to_repr();
            let a: Scalar = unsafe { std::mem::transmute(a_repr) };
            let b = Scalar::from_repr(a_repr).unwrap();
            assert_eq!(call_kernel("test_mont", &[GpuScalar(a)], &[]), b);
        }
    }
}
