# `gpu-tests`

This crate is for running tests. Usually kernels are created during compile time, hence a `build.rs` is needed. `ec-gpu-gen` is just a toolkit and doesn't provide pre-defined kernels. This crate separates those concerns and also shows how `ec-gpu-gen` can be used.

## Usage

```console
cargo test
```

## Feature flags

By default `cuda` and `opencl` is enabled. If you want to run the tests/benchmarks with either of those, you can do so:

```console
cargo test --no-default-features --features opencl
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
