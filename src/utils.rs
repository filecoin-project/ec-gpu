/// Divide anything into limbs of type `E`
pub fn limbs_of<T, E: Clone>(value: T) -> Vec<E> {
    unsafe {
        std::slice::from_raw_parts(
            &value as *const T as *const E,
            std::mem::size_of::<T>() / std::mem::size_of::<E>(),
        )
        .to_vec()
    }
}
