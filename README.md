# `ec-gpu` & `ec-gpu-gen`

[![crates.io][crate-image-ec-gpu]][crate-link-ec-gpu]
[![Documentation][doc-image-ec-gpu]][doc-link-ec-gpu]
[![Build Status][build-image-ec-gpu]][build-link-ec-gpu]
![minimum rustc 1.51][msrv-image-ec-gpu]
[![dependency status][deps-image-ec-gpu]][deps-link-ec-gpu]

[![crates.io][crate-image-ec-gpu-gen]][crate-link-ec-gpu-gen]
[![Documentation][doc-image-ec-gpu-gen]][doc-link-ec-gpu-gen]
[![Build Status][build-image-ec-gpu-gen]][build-link-ec-gpu-gen]
![minimum rustc 1.51][msrv-image-ec-gpu-gen]
[![dependency status][deps-image-ec-gpu-gen]][deps-link-ec-gpu-gen]

CUDA/OpenCL code generator for finite-field arithmetic over prime fields and elliptic curve arithmetic constructed with Rust.

Notes:
 - Limbs are 32/64-bit long, by your choice (on CUDA only 32-bit limbs are supported).
 - The library assumes that the most significant bit of your prime-field is unset. This allows for cheap reductions.

## Usage

### Quickstart

Generating CUDA/OpenCL codes for `blstrs` Scalar elements:

```rust
use blstrs::Scalar;
use ec_gpu_gen::SourceBuilder;

let source = SourceBuilder::new()
    .add_field::<Scalar>()
    .build_64_bit_limbs();
```

### Integration into your library

This crate usually creates GPU kernels at compile-time. CUDA generates a [fatbin], which OpenCL only generates the source code, which is then compiled at run-time.

In order to make things easier to use, there are helper functions available. You would put some code into `build.rs`, that generates the kernels, and some code into your library which then consumes those generated kernels. The kernels will be directly embedded into your program/library. If something goes wrong, you will get an error at compile-time.

In this example we will make use of the FFT functionality. Add to your `build.rs`:

```rust
use blstrs::Scalar;
use ec_gpu_gen::SourceBuilder;

fn main() {
    let source_builder = SourceBuilder::new().add_fft::<Scalar>()
    ec_gpu_gen::generate(&source_builder);
}
```

The `ec_gpu_gen::generate()` takes care of the actual code generation/compilation. It will automatically create a CUDA and/or OpenCL kernel. It will define two environment variables, which are meant for internal use. `_EC_GPU_CUDA_KERNEL_FATBIN` that points to the compiled CUDA kernel, and `_EC_GPU_OPENCL_KERNEL_SOURCE` that points to the generated OpenCL source.

Those variables are then picked up by the `ec_gpu_gen::program!()` macro, which generates a program, for a given GPU device. Using FFT within your library would then look like this:

```rust
use ec_gpu_gen::{
    rust_gpu_tools::Device,
};

let devices = Device::all();
let programs = devices
    .iter()
    .map(|device| ec_gpu_gen::program!(device))
    .collect::<Result<_, _>>()
    .expect("Cannot create programs!");

let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");
kern.radix_fft_many(&mut [&mut coeffs], &[omega], &[log_d]).expect("GPU FFT failed!");
```

## Feature flags

This crate supports CUDA and OpenCL, which can be enabled with the `cuda` and `opencl` feature flags.

### Environment variables

 - `EC_GPU_CUDA_NVCC_ARGS`

     By default the CUDA kernel is compiled for several architectures, which may take a long time. `EC_GPU_CUDA_NVCC_ARGS` can be used to override those arguments. The input and output file will still be automatically set.

    ```console
    // Example for compiling the kernel for only the Turing architecture.
    EC_GPU_CUDA_NVCC_ARGS="--fatbin --gpu-architecture=sm_75 --generate-code=arch=compute_75,code=sm_75"
    ```

 - `EC_GPU_FRAMEWORK`

    When the library is built with both CUDA and OpenCL support, you can choose which one to use at run time. The default is `cuda`, when you set nothing or any other (invalid) value. The other possible value is `opencl`.

    ```console
    // Example for setting it to OpenCL.
    EC_GPU_FRAMEWORK=opencl
    ```

 - `EC_GPU_NUM_THREADS`

   Restricts the number of threads used in the library. The default is set to the number of logical cores reported on the machine.

    ```console
    // Example for setting the maximum number of threads to 6.
    EC_GPU_NUM_THREADS=6
    ```


## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.


[crate-image-ec-gpu]: https://img.shields.io/crates/v/ec-gpu.svg
[crate-link-ec-gpu]: https://crates.io/crates/ec-gpu
[doc-image-ec-gpu]: https://docs.rs/ec-gpu/badge.svg
[doc-link-ec-gpu]: https://docs.rs/ec-gpu
[build-image-ec-gpu]: https://circleci.com/gh/filecoin-project/ec-gpu.svg?style=shield
[build-link-ec-gpu]: https://circleci.com/gh/filecoin-project/ec-gpu
[msrv-image-ec-gpu]: https://img.shields.io/badge/rustc-1.54+-blue.svg
[deps-image-ec-gpu]: https://deps.rs/repo/github/filecoin-projectt/ec-gpu/status.svg
[deps-link-ec-gpu]: https://deps.rs/repo/github/filecoin-project/ec-gpu


[crate-image-ec-gpu-gen]: https://img.shields.io/crates/v/ec-gpu-gen.svg
[crate-link-ec-gpu-gen]: https://crates.io/crates/ec-gpu-gen
[doc-image-ec-gpu-gen]: https://docs.rs/ec-gpu-gen/badge.svg
[doc-link-ec-gpu-gen]: https://docs.rs/ec-gpu-gen
[build-image-ec-gpu-gen]: https://circleci.com/gh/filecoin-project/ec-gpu.svg?style=shield
[build-link-ec-gpu-gen]: https://circleci.com/gh/filecoin-project/ec-gpu
[msrv-image-ec-gpu-gen]: https://img.shields.io/badge/rustc-1.54+-blue.svg
[deps-image-ec-gpu-gen]: https://deps.rs/repo/github/filecoin-projectt/ec-gpu/status.svg
[deps-link-ec-gpu-gen]: https://deps.rs/repo/github/filecoin-project/ec-gpu

[Fast Fourier transform]: https://en.wikipedia.org/wiki/Fast_Fourier_transform
[fatbin]: https://en.wikipedia.org/wiki/Fat_binary#Heterogeneous_computing
