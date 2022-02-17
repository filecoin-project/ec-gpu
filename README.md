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

Generating CUDA/OpenCL codes for `blstrs` Scalar elements:

```rust
use blstrs::Scalar;
let src = [
    ec_gpu_gen::common(),
    ec_gpu_gen::field::<Scalar, ec_gpu_gen::Limb64>("Fr")
].join("\n");
```
Generated interface (`FIELD` is substituted with `Fr`):

```c
#define FIELD_LIMB_BITS ... // 32/64
#define FIELD_limb ... // uint/ulong, based on FIELD_LIMB_BITS
#define FIELD_LIMBS ... // Number of limbs for this field
#define FIELD_P ... // Normal form of field modulus
#define FIELD_ONE ... // Montomery form of one
#define FIELD_ZERO ... // Montomery/normal form of zero
#define FIELD_BITS (FIELD_LIMBS * FIELD_LIMB_BITS)

typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;

bool FIELD_gte(FIELD a, FIELD b); // Greater than or equal
bool FIELD_eq(FIELD a, FIELD b); // Equal
FIELD FIELD_sub(FIELD a, FIELD b); // Modular subtraction
FIELD FIELD_add(FIELD a, FIELD b); // Modular addition
FIELD FIELD_mul(FIELD a, FIELD b); // Modular multiplication
FIELD FIELD_sqr(FIELD a); // Modular squaring
FIELD FIELD_double(FIELD a); // Modular doubling
FIELD FIELD_pow(FIELD base, uint exponent); // Modular power
FIELD FIELD_pow_lookup(global FIELD *bases, uint exponent); // Modular power with lookup table for bases
FIELD FIELD_mont(FIELD a); // To montgomery form
FIELD FIELD_unmont(FIELD a); // To regular form
bool FIELD_get_bit(FIELD l, uint i); // Get `i`th bit (From most significant digit)
uint FIELD_get_bits(FIELD l, uint skip, uint window); // Get `window` consecutive bits, (Starting from `skip`th bit from most significant digit)
```

## Feature flags

This crate contains implementation for [Fast Fourier transform] and multi-exponentiation, those are enabled be default by the `fft` and `multiexp` features. CPU and GPU implementations are available, you can enable CUDA and OpenCL support with the `cuda` and `opencl` feature flags.

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
