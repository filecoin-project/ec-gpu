//! An interface for dealing with the kinds of parallel computations involved.
//!
//! On native targets, uses a thread pool (yastl) for parallel computation.
//! On `wasm32` targets, executes everything on the main thread since WASM
//! does not support spawning OS threads.

// ════════════════════════════════════════════════════════════════
// Native implementation (multi-threaded via yastl)
// ════════════════════════════════════════════════════════════════

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::env;

    use crossbeam_channel::{bounded, Receiver, SendError};
    use log::trace;
    use once_cell::sync::Lazy;
    use yastl::Pool;

    static NUM_THREADS: Lazy<usize> = Lazy::new(read_num_threads);

    pub static THREAD_POOL: Lazy<Pool> = Lazy::new(|| Pool::new(*NUM_THREADS));

    fn read_num_threads() -> usize {
        env::var("EC_GPU_NUM_THREADS")
            .ok()
            .and_then(|num| num.parse::<usize>().ok())
            .unwrap_or_else(num_cpus::get)
    }

    #[derive(Clone, Default)]
    pub struct Worker {}

    impl Worker {
        pub fn new() -> Worker {
            Worker {}
        }

        pub fn log_num_threads(&self) -> u32 {
            log2_floor(*NUM_THREADS)
        }

        pub fn compute<F, R>(&self, f: F) -> Waiter<R>
        where
            F: FnOnce() -> R + Send + 'static,
            R: Send + 'static,
        {
            let (sender, receiver) = bounded(1);

            THREAD_POOL.spawn(move || {
                let res = f();
                if let Err(SendError(_)) = sender.send(res) {
                    trace!("Cannot send result");
                }
            });

            Waiter { receiver }
        }

        pub fn scope<'a, F, R>(&self, elements: usize, f: F) -> R
        where
            F: FnOnce(&yastl::Scope<'a>, usize) -> R,
        {
            let chunk_size = if elements < *NUM_THREADS {
                1
            } else {
                elements / *NUM_THREADS
            };

            THREAD_POOL.scoped(|scope| f(scope, chunk_size))
        }

        pub fn scoped<'a, F, R>(&self, f: F) -> R
        where
            F: FnOnce(&yastl::Scope<'a>) -> R,
        {
            let (sender, receiver) = bounded(1);
            THREAD_POOL.scoped(|s| {
                let res = f(s);
                sender.send(res).unwrap();
            });

            receiver.recv().unwrap()
        }
    }

    pub struct Waiter<T> {
        receiver: Receiver<T>,
    }

    impl<T> Waiter<T> {
        pub fn wait(&self) -> T {
            self.receiver.recv().unwrap()
        }

        pub fn done(val: T) -> Self {
            let (sender, receiver) = bounded(1);
            sender.send(val).unwrap();
            Waiter { receiver }
        }
    }

    pub(crate) fn log2_floor(num: usize) -> u32 {
        assert!(num > 0);
        let mut pow = 0;
        while (1 << (pow + 1)) <= num {
            pow += 1;
        }
        pow
    }
}

// ════════════════════════════════════════════════════════════════
// WASM implementation (single-threaded fallback)
// ════════════════════════════════════════════════════════════════

#[cfg(target_arch = "wasm32")]
mod wasm {
    /// A worker that executes everything on the main thread.
    ///
    /// WASM does not support spawning OS threads (std::thread::spawn panics).
    /// This fallback runs all computations sequentially on the calling thread.
    /// Performance is lower than native multi-threaded execution but correctness
    /// is preserved — all bellperson operations are safe to run single-threaded.
    #[derive(Clone, Default)]
    pub struct Worker {}

    impl Worker {
        pub fn new() -> Worker {
            Worker {}
        }

        /// Returns 0 (single-threaded = 2^0 = 1 thread).
        pub fn log_num_threads(&self) -> u32 {
            0
        }

        /// Executes the function immediately on the current thread.
        pub fn compute<F, R>(&self, f: F) -> Waiter<R>
        where
            F: FnOnce() -> R + Send + 'static,
            R: Send + 'static,
        {
            let result = f();
            Waiter { value: std::cell::RefCell::new(Some(result)) }
        }

        /// Executes with chunk_size = elements (single thread processes everything).
        pub fn scope<'a, F, R>(&self, elements: usize, f: F) -> R
        where
            F: FnOnce(&WasmScope<'a>, usize) -> R,
        {
            let scope = WasmScope::new();
            f(&scope, elements)
        }

        /// Executes the function immediately.
        pub fn scoped<'a, F, R>(&self, f: F) -> R
        where
            F: FnOnce(&WasmScope<'a>) -> R,
        {
            let scope = WasmScope::new();
            f(&scope)
        }
    }

    /// A minimal scope that executes spawned closures immediately (sequentially).
    ///
    /// Mimics the `yastl::Scope<'a>` API but runs everything on the current thread.
    /// The lifetime parameter matches yastl::Scope's signature for API compatibility.
    pub struct WasmScope<'a> {
        _marker: std::marker::PhantomData<&'a ()>,
    }

    impl<'a> WasmScope<'a> {
        fn new() -> Self {
            WasmScope { _marker: std::marker::PhantomData }
        }

        /// "Spawns" a closure by executing it immediately on the current thread.
        pub fn execute<F>(&self, f: F)
        where
            F: FnOnce() + Send,
        {
            f();
        }
    }

    /// A future that already has its value (computed synchronously).
    pub struct Waiter<T> {
        value: std::cell::RefCell<Option<T>>,
    }

    impl<T> Waiter<T> {
        /// Takes the value. Panics if called more than once.
        pub fn wait(&self) -> T {
            self.value.borrow_mut().take()
                .expect("Waiter::wait called on already-consumed waiter")
        }

        pub fn done(val: T) -> Self {
            Waiter { value: std::cell::RefCell::new(Some(val)) }
        }
    }

    /// A single-threaded "pool" for WASM — executes closures inline.
    pub struct WasmPool;

    impl WasmPool {
        pub fn scoped<'a, F, R>(&self, f: F) -> R
        where
            F: FnOnce(&WasmScope<'a>) -> R,
        {
            let scope = WasmScope::new();
            f(&scope)
        }
    }

    /// Global "thread pool" for WASM — single-threaded.
    pub static THREAD_POOL: WasmPool = WasmPool;

    pub(crate) fn log2_floor(num: usize) -> u32 {
        assert!(num > 0);
        let mut pow = 0;
        while (1 << (pow + 1)) <= num {
            pow += 1;
        }
        pow
    }
}

// ════════════════════════════════════════════════════════════════
// Re-exports — callers use Worker, Waiter, THREAD_POOL without
// knowing which implementation is active.
// ════════════════════════════════════════════════════════════════

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_floor() {
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(3), 1);
        assert_eq!(log2_floor(4), 2);
        assert_eq!(log2_floor(5), 2);
        assert_eq!(log2_floor(6), 2);
        assert_eq!(log2_floor(7), 2);
        assert_eq!(log2_floor(8), 3);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_read_num_threads() {
        let num_cpus = num_cpus::get();
        temp_env::with_var("EC_GPU_NUM_THREADS", None::<&str>, || {
            assert_eq!(
                native::read_num_threads(),
                num_cpus,
                "By default the number of threads matches the number of CPUs."
            );
        });

        temp_env::with_var("EC_GPU_NUM_THREADS", Some("1234"), || {
            assert_eq!(
                native::read_num_threads(),
                1234,
                "Number of threads matches the environment variable."
            );
        });
    }
}
