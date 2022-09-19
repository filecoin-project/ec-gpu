#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    use blstrs::{Fp, Fp2, G1Affine, G2Affine, Scalar};
    use ec_gpu_gen::SourceBuilder;

    let source_builder = SourceBuilder::new()
        .add_fft::<Scalar>()
        .add_multiexp::<G1Affine, Fp>()
        .add_multiexp::<G2Affine, Fp2>();
    ec_gpu_gen::generate(&source_builder);
}
