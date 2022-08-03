/// The name that is used in the GPU source code to identify the item that is used.
pub trait GpuName {
    /// A unique name for the item.
    ///
    /// To make the uniqueness easier to implement, use the [`name`] macro. It produces a unique
    /// name, based on the module path and the type of the item itself. That identifier might not
    /// be stable across different versions of a crate, but this is OK as kernel sources/binaries
    /// are always bundled with a library and not re-used between versions.
    ///
    /// # Example
    ///
    /// ```
    /// struct Fp;
    ///
    /// impl ec_gpu::GpuName for Fp {
    ///     fn name() -> String {
    ///         ec_gpu::name!()
    ///     }
    /// }
    /// ```
    fn name() -> String;
}

/// A prime field that returns the values in a representation that is suited for the use on a GPU.
pub trait GpuField: GpuName {
    /// Returns `1` as a vector of 32-bit limbs in little-endian non-Montgomery form (least
    /// significant limb first).
    fn one() -> Vec<u32>;

    /// Returns `R ^ 2 mod P` as a vector of 32-bit limbs in little-endian non-Montgomery form
    /// (least significant limb first).
    fn r2() -> Vec<u32>;

    /// Returns the field modulus as a vector of 32-bit limbs in non-Montgomery form (least
    /// significant limb first).
    fn modulus() -> Vec<u32>;

    /// If the field is an extension field, then the name of the sub-field is returned.
    fn sub_field_name() -> Option<String> {
        None
    }
}

/// Macro to get a unique name of an item.
///
/// The name is a string that consists of the module path and the type name. All non-alphanumeric
/// characters are replaced with underscores, so that it's an identifier that doesn't cause any
/// issues with C compilers.
#[macro_export]
macro_rules! name {
    () => {{
        let mod_path = module_path!();
        let type_name = core::any::type_name::<Self>();
        let name = if type_name.starts_with(mod_path) {
            type_name.into()
        } else {
            [mod_path, "__", type_name].concat()
        };
        name.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
    }};
}
