use std::cmp;
use std::ops::MulAssign;
use std::sync::{Arc, RwLock};

use ec_gpu::GpuEngine;
use ff::Field;
use log::{error, info};
use pairing::Engine;
use rust_gpu_tools::{program_closures, Device, LocalBuffer, Program};

use crate::threadpool::THREAD_POOL;
use crate::{
    error::{EcError, EcResult},
    program,
};

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

/// FFT kernel for a single GPU.
pub struct SingleFftKernel<'a, E>
where
    E: Engine + GpuEngine,
{
    program: Program,
    /// An optional function which will be called at places where it is possible to abort the FFT
    /// calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    _phantom: std::marker::PhantomData<E::Fr>,
}

impl<'a, E: Engine + GpuEngine> SingleFftKernel<'a, E> {
    /// Create a new kernel for a device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        device: &Device,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let program = program::program(device)?;

        Ok(SingleFftKernel {
            program,
            maybe_abort,
            _phantom: Default::default(),
        })
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(&mut self, input: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> EcResult<()> {
        let closures = program_closures!(|program, input: &mut [E::Fr]| -> EcResult<()> {
            let n = 1 << log_n;
            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            let mut dst_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
            let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = E::Fr::one();
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            let pq_buffer = program.create_buffer_from_slice(&pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![E::Fr::zero(); 32];
            omegas[0] = *omega;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            program.write_from_buffer(&mut src_buffer, &*input)?;
            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round
            while log_p < log_n {
                if let Some(maybe_abort) = &self.maybe_abort {
                    if maybe_abort() {
                        return Err(EcError::Aborted);
                    }
                }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "radix_fft",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            program.read_into_buffer(&src_buffer, input)?;

            Ok(())
        });

        self.program.run(closures, input)
    }
}

/// One FFT kernel for each GPU available.
pub struct FftKernel<'a, E>
where
    E: Engine + GpuEngine,
{
    kernels: Vec<SingleFftKernel<'a, E>>,
}

impl<'a, E> FftKernel<'a, E>
where
    E: Engine + GpuEngine,
{
    /// Create new kernels, one for each given device.
    pub fn create(devices: &[&Device]) -> EcResult<Self> {
        Self::create_optional_abort(devices, None)
    }

    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create_with_abort(
        devices: &[&Device],
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        Self::create_optional_abort(devices, Some(maybe_abort))
    }

    fn create_optional_abort(
        devices: &[&Device],
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let kernels: Vec<_> = devices
            .iter()
            .filter_map(|device| {
                let kernel = SingleFftKernel::<E>::create(device, maybe_abort);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(EcError::Simple("No working GPUs found!"));
        }
        info!("FFT: {} working device(s) selected. ", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!("FFT: Device {}: {}", i, k.program.device_name(),);
        }

        Ok(Self { kernels })
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses the first available GPU.
    pub fn radix_fft(&mut self, input: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> EcResult<()> {
        self.kernels[0].radix_fft(input, omega, log_n)
    }

    /// Performs FFT on `inputs`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses all available GPUs to distribute the work.
    pub fn radix_fft_many(
        &mut self,
        inputs: &mut [&mut [E::Fr]],
        omegas: &[E::Fr],
        log_ns: &[u32],
    ) -> EcResult<()> {
        let n = inputs.len();
        let num_devices = self.kernels.len();
        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

        let result = Arc::new(RwLock::new(Ok(())));

        THREAD_POOL.scoped(|s| {
            for (((inputs, omegas), log_ns), kern) in inputs
                .chunks_mut(chunk_size)
                .zip(omegas.chunks(chunk_size))
                .zip(log_ns.chunks(chunk_size))
                .zip(self.kernels.iter_mut())
            {
                let result = result.clone();
                s.execute(move || {
                    for ((input, omega), log_n) in
                        inputs.iter_mut().zip(omegas.iter()).zip(log_ns.iter())
                    {
                        if result.read().unwrap().is_err() {
                            break;
                        }

                        if let Err(err) = kern.radix_fft(input, omega, *log_n) {
                            *result.write().unwrap() = Err(err);
                            break;
                        }
                    }
                });
            }
        });

        Arc::try_unwrap(result).unwrap().into_inner().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use blstrs::{Bls12, Scalar as Fr};
    use ff::{Field, PrimeField};
    use std::time::Instant;

    use crate::fft_cpu::{parallel_fft, serial_fft};
    use crate::threadpool::Worker;

    fn omega<E: Engine>(num_coeffs: usize) -> E::Fr {
        // Compute omega, the 2^exp primitive root of unity
        let exp = (num_coeffs as f32).log2().floor() as u32;
        let mut omega = E::Fr::root_of_unity();
        for _ in exp..E::Fr::S {
            omega = omega.square();
        }
        omega
    }

    #[test]
    pub fn gpu_fft_consistency() {
        let mut rng = rand::thread_rng();

        let worker = Worker::new();
        let log_threads = worker.log_num_threads();
        let devices = Device::all();
        let mut kern = FftKernel::<Bls12>::create(&devices).expect("Cannot initialize kernel!");

        for log_d in 1..=20 {
            let d = 1 << log_d;

            let mut v1_coeffs = (0..d).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            let v1_omega = omega::<Bls12>(v1_coeffs.len());
            let mut v2_coeffs = v1_coeffs.clone();
            let v2_omega = v1_omega;

            println!("Testing FFT for {} elements...", d);

            let mut now = Instant::now();
            kern.radix_fft_many(&mut [&mut v1_coeffs], &[v1_omega], &[log_d])
                .expect("GPU FFT failed!");
            let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("GPU took {}ms.", gpu_dur);

            now = Instant::now();
            if log_d <= log_threads {
                serial_fft::<Bls12>(&mut v2_coeffs, &v2_omega, log_d);
            } else {
                parallel_fft::<Bls12>(&mut v2_coeffs, &worker, &v2_omega, log_d, log_threads);
            }
            let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("CPU ({} cores) took {}ms.", 1 << log_threads, cpu_dur);

            println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

            assert!(v1_coeffs == v2_coeffs);
            println!("============================");
        }
    }

    #[test]
    pub fn gpu_fft_many_consistency() {
        let mut rng = rand::thread_rng();

        let worker = Worker::new();
        let log_threads = worker.log_num_threads();
        let devices = Device::all();
        let mut kern = FftKernel::<Bls12>::create(&devices).expect("Cannot initialize kernel!");

        for log_d in 1..=20 {
            let d = 1 << log_d;

            let mut v11_coeffs = (0..d).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            let mut v12_coeffs = (0..d).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            let mut v13_coeffs = (0..d).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            let v11_omega = omega::<Bls12>(v11_coeffs.len());
            let v12_omega = omega::<Bls12>(v12_coeffs.len());
            let v13_omega = omega::<Bls12>(v13_coeffs.len());

            let mut v21_coeffs = v11_coeffs.clone();
            let mut v22_coeffs = v12_coeffs.clone();
            let mut v23_coeffs = v13_coeffs.clone();
            let v21_omega = v11_omega;
            let v22_omega = v12_omega;
            let v23_omega = v13_omega;

            println!("Testing FFT3 for {} elements...", d);

            let mut now = Instant::now();
            kern.radix_fft_many(
                &mut [&mut v11_coeffs, &mut v12_coeffs, &mut v13_coeffs],
                &[v11_omega, v12_omega, v13_omega],
                &[log_d, log_d, log_d],
            )
            .expect("GPU FFT failed!");
            let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("GPU took {}ms.", gpu_dur);

            now = Instant::now();
            if log_d <= log_threads {
                serial_fft::<Bls12>(&mut v21_coeffs, &v21_omega, log_d);
                serial_fft::<Bls12>(&mut v22_coeffs, &v22_omega, log_d);
                serial_fft::<Bls12>(&mut v23_coeffs, &v23_omega, log_d);
            } else {
                parallel_fft::<Bls12>(&mut v21_coeffs, &worker, &v21_omega, log_d, log_threads);
                parallel_fft::<Bls12>(&mut v22_coeffs, &worker, &v22_omega, log_d, log_threads);
                parallel_fft::<Bls12>(&mut v23_coeffs, &worker, &v23_omega, log_d, log_threads);
            }
            let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("CPU ({} cores) took {}ms.", 1 << log_threads, cpu_dur);

            println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

            assert!(v11_coeffs == v21_coeffs);
            assert!(v12_coeffs == v22_coeffs);
            assert!(v13_coeffs == v23_coeffs);

            println!("============================");
        }
    }
}
