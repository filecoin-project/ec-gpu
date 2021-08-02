#[cfg(test)]
mod tests {
    use ff_cl_gen::*;

    use blstrs::Scalar as Fr;
    use ff::{Field, PrimeField};
    use lazy_static::lazy_static;
    use ocl::{OclPrm, ProQue};
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

    static TEST_SRC: &str = include_str!("test.cl");

    lazy_static! {
        static ref PROQUE: ProQue = {
            let src = vec![
                field::<Fr, Limb32>("Fr32"),
                field::<Fr, Limb64>("Fr64"),
                TEST_SRC.to_string(),
            ]
            .join("\n\n");

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
            let c = a + b;

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
            let c = a - b;
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
            let c = a * b;

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
            let c = a.pow_vartime([b as u64]);
            assert_eq!(call_kernel!("test_pow_32", GpuFr(a), b), c);
            assert_eq!(call_kernel!("test_pow_64", GpuFr(a), b), c);
        }
    }

    #[test]
    fn test_sqr() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = a.square();

            assert_eq!(call_kernel!("test_sqr_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_sqr_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b = a.double();

            assert_eq!(call_kernel!("test_double_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_double_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_unmont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::random(&mut rng);
            let b: Fr = unsafe { std::mem::transmute(a.to_repr()) };
            assert_eq!(call_kernel!("test_unmont_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_unmont_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_mont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a_repr = Fr::random(&mut rng).to_repr();
            let a: Fr = unsafe { std::mem::transmute(a_repr) };
            let b = Fr::from_repr(a_repr).unwrap();
            assert_eq!(call_kernel!("test_mont_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_mont_64", GpuFr(a)), b);
        }
    }
}
