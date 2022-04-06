#![warn(missing_docs)]
//! CUDA/OpenCL code generator for finite-field arithmetic over prime fields and elliptic curve
//! arithmetic constructed with Rust.
//!
//! There is also support for Fast Fourier Transform and Multiexponentiation.
//!
//! This crate usually creates GPU kernels at compile-time. CUDA generates a [fatbin], which OpenCL only generates the source code, which is then compiled at run-time.
//!
//! In order to make things easier to use, there are helper functions available. You would put some code into `build.rs`, that generates the kernels, and some code into your library which then consumes those generated kernels. The kernels will be directly embedded into your program/library. If something goes wrong, you will get an error at compile-time.
//!
//! In this example we will make use of the FFT functionality. Add to your `build.rs`:
//!
//! ```no_run
//! use blstrs::Scalar;
//! use ec_gpu_gen::SourceBuilder;
//!
//! let source_builder = SourceBuilder::new().add_fft::<Scalar>();
//! ec_gpu_gen::generate(&source_builder);
//! ```
//!
//! The `ec_gpu_gen::generate()` takes care of the actual code generation/compilation. It will automatically create a CUDA and/or OpenCL kernel. It will define two environment variables, which are meant for internal use. `_EC_GPU_CUDA_KERNEL_FATBIN` that points to the compiled CUDA kernel, and `_EC_GPU_OPENCL_KERNEL_SOURCE` that points to the generated OpenCL source.
//!
//! Those variables are then picked up by the `ec_gpu_gen::program!()` macro, which generates a program, for a given GPU device. Using FFT within your library would then look like this:
//!
//! ```no_compile
//! use blstrs::Scalar;
//! use ec_gpu_gen::{
//!     rust_gpu_tools::Device,
//! };
//!
//! let devices = Device::all();
//! let programs = devices
//!     .iter()
//!     .map(|device| ec_gpu_gen::program!(device))
//!     .collect::<Result<_, _>>()
//!     .expect("Cannot create programs!");
//!
//! let mut kern = FftKernel::<Scalar>::create(programs).expect("Cannot initialize kernel!");
//! kern.radix_fft_many(&mut [&mut coeffs], &[omega], &[log_d]).expect("GPU FFT failed!");
//! ```
//!
//! Feature flags
//! -------------
//!
//! CUDA and OpenCL are supprted, each be enabled with the `cuda` and `opencl` [feature flags].
//!
//! [fatbin]: https://en.wikipedia.org/wiki/Fat_binary#Heterogeneous_computing
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section
mod error;
#[cfg(any(feature = "cuda", feature = "opencl"))]
mod program;
mod source;

/// Fast Fourier Transform on the GPU.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub mod fft;
/// Fast Fourier Transform on the CPU.
pub mod fft_cpu;
/// Multiexponentiation on the GPU.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub mod multiexp;
/// Multiexponentiation on the CPU.
pub mod multiexp_cpu;
/// Helpers for multithreaded code.
pub mod threadpool;

/// Re-export rust-gpu-tools as things like [`rust_gpu_tools::Device`] might be needed.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use rust_gpu_tools;

pub use error::{EcError, EcResult};
pub use source::{generate, SourceBuilder};
