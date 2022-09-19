#[macro_export]
/// Helper macro to create a program for a device.
///
/// It will embed the CUDA fatbin/OpenCL source code within your binary. The source needs to be
/// generated via [`crate::source::generate`] in your `build.rs`.
///
/// It returns a `[crate::rust_gpu_tools::Program`] instance.
macro_rules! program {
    ($device:ident) => {{
        use $crate::rust_gpu_tools::{Framework, GPUError, Program};
        (|device: &Device| -> Result<Program, $crate::EcError> {
            // Selects a CUDA or OpenCL on the `EC_GPU_FRAMEWORK` environment variable and the
            // compile-time features.
            //
            // You cannot select CUDA if the library was compiled without support for it.
            let default_framework = device.framework();
            let framework = match ::std::env::var("EC_GPU_FRAMEWORK") {
                Ok(env) => match env.as_ref() {
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        {
                            Framework::Cuda
                        }

                        #[cfg(not(feature = "cuda"))]
                        return Err($crate::EcError::Simple("CUDA framework is not supported, please compile with the `cuda` feature enabled."))
                    }
                    "opencl" => {
                        #[cfg(feature = "opencl")]
                        {
                            Framework::Opencl
                        }

                        #[cfg(not(feature = "opencl"))]
                        return Err($crate::EcError::Simple("OpenCL framework is not supported, please compile with the `opencl` feature enabled."))
                    }
                    _ => default_framework,
                },
                Err(_) => default_framework,
            };

            match framework {
                #[cfg(feature = "cuda")]
                Framework::Cuda => {
                    let kernel = include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));
                    let cuda_device = device.cuda_device().ok_or(GPUError::DeviceNotFound)?;
                    let program = $crate::rust_gpu_tools::cuda::Program::from_bytes(cuda_device, kernel)?;
                    Ok(Program::Cuda(program))
                }
                #[cfg(feature = "opencl")]
                Framework::Opencl => {
                    let source = include_str!(env!("_EC_GPU_OPENCL_KERNEL_SOURCE"));
                    let opencl_device = device.opencl_device().ok_or(GPUError::DeviceNotFound)?;
                    let program = $crate::rust_gpu_tools::opencl::Program::from_opencl(opencl_device, source)?;
                    Ok(Program::Opencl(program))
                }
            }
        })($device)
    }};
}
