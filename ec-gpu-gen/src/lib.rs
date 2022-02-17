#![warn(missing_docs)]
//! CUDA/OpenCL code generator for finite-field arithmetic over prime fields and elliptic curve
//! arithmetic constructed with Rust.
//!
//! There is also support for Fast Fourier Transform and Multiexponentiation.
//!
//!
//! Feature flags
//! -------------
//!
//! There are several [feature flags] that can be combined in all possible ways. By default, all
//! features are enabled. You can enable CUDA and/or OpenCL support with the `cuda` and the
//! `opencl` features. Those can be combined with the `fft` and the `multiexp` feature. If one of
//! them is enabled, a kernel with that functionality will be generated.
//!
//! If you only want to use the CPU version of FFT and/or multiexp you can do so by enabling the
//! `fft` and/or `multiexp`, and disabling the `cuda` and `opencl` features.
//!
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section
mod error;
#[cfg(any(feature = "cuda", feature = "opencl"))]
mod program;
mod source;

/// Fast Fourier Transform on the GPU.
#[cfg(all(feature = "fft", any(feature = "cuda", feature = "opencl")))]
pub mod fft;
/// Fast Fourier Transform on the CPU.
#[cfg(feature = "fft")]
pub mod fft_cpu;
/// Multiexponentiation on the GPU.
#[cfg(all(feature = "multiexp", any(feature = "cuda", feature = "opencl")))]
pub mod multiexp;
/// Multiexponentiation on the CPU.
#[cfg(feature = "multiexp")]
pub mod multiexp_cpu;
/// Helpers for multithreaded code.
#[cfg(any(feature = "fft", feature = "multiexp"))]
pub mod threadpool;

/// Re-export rust-gpu-tools as things like [`rust_gpu_tools::Device`] might be needed.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use rust_gpu_tools;

pub use error::{EcError, EcResult};
pub use source::{common, field, gen_ec_source, gen_source, Limb, Limb32, Limb64};
