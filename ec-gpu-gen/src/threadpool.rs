//! An interface for dealing with the kinds of parallel computations involved.
use std::env;

use crossbeam_channel::{bounded, Receiver, SendError};
use log::trace;
use once_cell::sync::Lazy;
use yastl::Pool;

/// The number of threads the thread pool should use.
///
/// By default it's equal to the number of CPUs, but it can be changed with the
/// `EC_GPU_NUM_THREADS` environment variable.
static NUM_THREADS: Lazy<usize> = Lazy::new(read_num_threads);

/// The thread pool that is used for the computations.
///
/// By default, it's size is equal to the number of CPUs. It can be set to a different value with
/// the `EC_GPU_NUM_THREADS` environment variable.
pub static THREAD_POOL: Lazy<Pool> = Lazy::new(|| Pool::new(*NUM_THREADS));

/// Returns the number of threads.
///
/// The number can be set with the `EC_GPU_NUM_THREADS` environment variable. If it isn't set, it
/// defaults to the number of CPUs the system has.
fn read_num_threads() -> usize {
    env::var("EC_GPU_NUM_THREADS")
        .ok()
        .and_then(|num| num.parse::<usize>().ok())
        .unwrap_or_else(num_cpus::get)
}

/// A worker operates on a pool of threads.
#[derive(Clone, Default)]
pub struct Worker {}

impl Worker {
    /// Returns a new worker.
    pub fn new() -> Worker {
        Worker {}
    }

    /// Returns binary logarithm (floored) of the number of threads.
    ///
    /// This means, the number of threads is `2^log_num_threads()`.
    pub fn log_num_threads(&self) -> u32 {
        log2_floor(*NUM_THREADS)
    }

    /// Executes a function in a thread and returns a [`Waiter`] immediately.
    pub fn compute<F, R>(&self, f: F) -> Waiter<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (sender, receiver) = bounded(1);

        THREAD_POOL.spawn(move || {
            let res = f();
            // Best effort. We run it in a separate thread, so the receiver might not exist
            // anymore, but that's OK. It only means that we are not interested in the result.
            // A message is logged though, as concurrency issues are hard to debug and this might
            // help in such cases.
            if let Err(SendError(_)) = sender.send(res) {
                trace!("Cannot send result");
            }
        });

        Waiter { receiver }
    }

    /// Executes a function and returns the result once it is finished.
    ///
    /// The function gets the [`yastl::Scope`] as well as the `chunk_size` as parameters. THe
    /// `chunk_size` is number of elements per thread.
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

    /// Executes the passed in function, and returns the result once it is finished.
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

/// A future that is waiting for a result.
pub struct Waiter<T> {
    receiver: Receiver<T>,
}

impl<T> Waiter<T> {
    /// Wait for the result.
    pub fn wait(&self) -> T {
        self.receiver.recv().unwrap()
    }

    /// One off sending.
    pub fn done(val: T) -> Self {
        let (sender, receiver) = bounded(1);
        sender.send(val).unwrap();

        Waiter { receiver }
    }
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

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

    #[test]
    fn test_read_num_threads() {
        let num_cpus = num_cpus::get();
        temp_env::with_var("EC_GPU_NUM_THREADS", None::<&str>, || {
            assert_eq!(
                read_num_threads(),
                num_cpus,
                "By default the number of threads matches the number of CPUs."
            );
        });

        temp_env::with_var("EC_GPU_NUM_THREADS", Some("1234"), || {
            assert_eq!(
                read_num_threads(),
                1234,
                "Number of threads matches the environment variable."
            );
        });
    }
}
