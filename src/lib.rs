use ff::PrimeField;
use itertools::*;
use num_bigint::BigUint;

pub trait Limb: Sized + Clone + Copy {
    type LimbType: Clone + std::fmt::Display;
    fn zero() -> Self;
    fn new(val: Self::LimbType) -> Self;
    fn value(&self) -> Self::LimbType;
    fn bits() -> usize;
    fn ptx_types() -> (&'static str, &'static str);
    fn opencl_type() -> &'static str;
    fn limbs_of<T>(value: T) -> Vec<Self> {
        limbs_of::<T, Self::LimbType>(value)
            .into_iter()
            .map(|l| Self::new(l))
            .collect()
    }
    /// Calculate the `INV` parameter of Montgomery reduction algorithm for 32/64bit limbs
    /// * `a` - Is the first limb of modulus
    fn calc_inv(a: Self) -> Self;
    fn calculate_r2<F: PrimeField>() -> Vec<Self>;
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
    fn ptx_types() -> (&'static str, &'static str) {
        ("u32", "r")
    }
    fn opencl_type() -> &'static str {
        "uint"
    }
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u32;
        for _ in 0..31 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn calculate_r2<F: PrimeField>() -> Vec<Self> {
        calculate_r2::<F>()
            .into_iter()
            .map(|l| Self::new(l))
            .collect()
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
    fn ptx_types() -> (&'static str, &'static str) {
        ("u64", "l")
    }
    fn opencl_type() -> &'static str {
        "ulong"
    }
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn calculate_r2<F: PrimeField>() -> Vec<Self> {
        calculate_r2::<F>()
            .into_iter()
            .tuples()
            .map(|(lo, hi)| Self::new(((hi as u64) << 32) + (lo as u64)))
            .collect()
    }
}

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");

/// Divide anything into limbs of type `E`
fn limbs_of<T, E: Clone>(value: T) -> Vec<E> {
    unsafe {
        std::slice::from_raw_parts(
            &value as *const T as *const E,
            std::mem::size_of::<T>() / std::mem::size_of::<E>(),
        )
        .to_vec()
    }
}

fn define_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "#define {} ((FIELD){{ {{ {} }} }})",
        name,
        join(limbs.iter().map(|l| l.value()), ", ")
    )
}

/// Calculates `R ^ 2 mod P` and returns the result as a vector of 64bit limbs
fn calculate_r2<F: PrimeField>() -> Vec<u32> {
    // R ^ 2 mod P
    BigUint::new(limbs_of::<_, u32>(F::one()))
        .modpow(
            &BigUint::from_slice(&[2]),                   // ^ 2
            &BigUint::new(limbs_of::<_, u32>(F::char())), // mod P
        )
        .to_u32_digits()
}

/// Generates OpenCL constants and type definitions of prime-field `F`
fn params<F, L: Limb>() -> String
where
    F: PrimeField,
{
    let one = L::limbs_of(F::one()); // Get Montgomery form of F::one()
    let p = L::limbs_of(F::char()); // Get regular form of field modulus
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
    let typedef = format!("typedef struct {{ FIELD_limb val[FIELD_LIMBS]; }} FIELD;");
    join(
        &[
            limb_def,
            limbs_def,
            limb_bits_def,
            one_def,
            p_def,
            r2_def,
            zero_def,
            inv_def,
            typedef,
        ],
        "\n",
    )
}

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
fn field_add_sub_nvidia<F, L: Limb>() -> String
where
    F: PrimeField,
{
    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_types();

    result.push_str("#ifdef NVIDIA\n");
    for op in &["sub", "add"] {
        let len = L::limbs_of(F::one()).len();

        let mut src = format!("FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{\n", op);
        if len > 1 {
            src.push_str("asm(");
            src.push_str(format!("\"{}.cc.{} %0, %0, %{};\\r\\n\"\n", op, ptx_type, len).as_str());
            for i in 1..len - 1 {
                src.push_str(
                    format!(
                        "\"{}c.cc.{} %{}, %{}, %{};\\r\\n\"\n",
                        op,
                        ptx_type,
                        i,
                        i,
                        len + i
                    )
                    .as_str(),
                );
            }
            src.push_str(
                format!(
                    "\"{}c.{} %{}, %{}, %{};\\r\\n\"\n",
                    op,
                    ptx_type,
                    len - 1,
                    len - 1,
                    2 * len - 1
                )
                .as_str(),
            );
            src.push_str(":");
            let inps = join(
                (0..len).map(|n| format!("\"+{}\"(a.val[{}])", ptx_reg, n)),
                ", ",
            );
            src.push_str(inps.as_str());

            src.push_str("\n:");
            let outs = join(
                (0..len).map(|n| format!("\"{}\"(b.val[{}])", ptx_reg, n)),
                ", ",
            );
            src.push_str(outs.as_str());
            src.push_str(");\n");
        }
        src.push_str("return a;\n}\n");

        result.push_str(&src);
    }
    result.push_str("#endif\n");

    result
}

/// Returns OpenCL source-code of a ff::PrimeField with name `name`
/// Find details in README.md
pub fn field<F, L: Limb>(name: &str) -> String
where
    F: PrimeField,
{
    join(
        &[
            COMMON_SRC.to_string(),
            params::<F, L>(),
            field_add_sub_nvidia::<F, L>(),
            String::from(FIELD_SRC),
        ],
        "\n",
    )
    .replace("FIELD", name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use lazy_static::lazy_static;
    use ocl::{OclPrm, ProQue};
    use paired::bls12_381::{Fr, FrRepr};
    use rand::{thread_rng, Rng};

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(transparent)]
    pub struct GpuFr(pub Fr);
    impl Default for GpuFr {
        fn default() -> Self {
            Self(Fr::zero())
        }
    }
    unsafe impl OclPrm for GpuFr {}

    lazy_static! {
        static ref PROQUE: ProQue = {
            static TEST_SRC: &str = include_str!("cl/test.cl");
            let src = format!(
                "{}\n{}\n{}",
                field::<Fr, Limb32>("Fr32"),
                field::<Fr, Limb64>("Fr64"),
                TEST_SRC
            );
            ProQue::builder().src(src).dims(1).build().unwrap()
        };
    }

    macro_rules! call_kernel {
        ($name:expr, $($arg:expr),*) => {{
            let mut cpu_buffer = vec![GpuFr::default()];
            let buffer = PROQUE.create_buffer::<GpuFr>().unwrap();
            buffer.write(&cpu_buffer).enq().unwrap();
            let kernel =
                PROQUE
                .kernel_builder($name)
                $(.arg($arg))*
                .arg(&buffer)
                .build().unwrap();
            unsafe {
                kernel.enq().unwrap();
            }
            buffer.read(&mut cpu_buffer).enq().unwrap();

            cpu_buffer[0].0
        }};
    }

    #[test]
    fn test_add() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let mut c = a.clone();
            c.add_assign(&b);
            assert_eq!(call_kernel!("test_add_32", GpuFr(a), GpuFr(b)), c);
            assert_eq!(call_kernel!("test_add_64", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let mut c = a.clone();
            c.sub_assign(&b);
            assert_eq!(call_kernel!("test_sub_32", GpuFr(a), GpuFr(b)), c);
            assert_eq!(call_kernel!("test_sub_64", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = Fr::random(&mut rng);
            let mut c = a.clone();
            c.mul_assign(&b);
            assert_eq!(call_kernel!("test_mul_32", GpuFr(a), GpuFr(b)), c);
            assert_eq!(call_kernel!("test_mul_64", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_pow() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = rng.gen::<u32>();
            let c = a.pow([b as u64]);
            assert_eq!(call_kernel!("test_pow_32", GpuFr(a), b), c);
            assert_eq!(call_kernel!("test_pow_64", GpuFr(a), b), c);
        }
    }

    #[test]
    fn test_sqr() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let mut b = a.clone();
            b.square();
            assert_eq!(call_kernel!("test_sqr_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_sqr_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let mut b = a.clone();
            b.double();
            assert_eq!(call_kernel!("test_double_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_double_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_unmont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = unsafe { std::mem::transmute::<FrRepr, Fr>(a.into_repr()) };
            assert_eq!(call_kernel!("test_unmont_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_unmont_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_mont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a_repr = Fr::random(&mut rng).into_repr();
            let a = unsafe { std::mem::transmute::<FrRepr, Fr>(a_repr) };
            let b = Fr::from_repr(a_repr).unwrap();
            assert_eq!(call_kernel!("test_mont_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_mont_64", GpuFr(a)), b);
        }
    }
}
