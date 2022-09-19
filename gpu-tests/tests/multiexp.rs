#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::sync::Arc;
use std::time::Instant;

use blstrs::Bls12;
use ec_gpu::GpuName;
use ec_gpu_gen::multiexp_cpu::{multiexp_cpu, FullDensity, QueryDensity, SourceBuilder};
use ec_gpu_gen::{
    multiexp::MultiexpKernel, program, rust_gpu_tools::Device, threadpool::Worker, EcError,
};
use ff::{Field, PrimeField};
use group::Curve;
use group::{prime::PrimeCurveAffine, Group};
use pairing::Engine;

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
