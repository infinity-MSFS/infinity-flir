//! Row-parallel loops on std threads (no thread-pool dependency).
//!
//! Every per-pixel stage of the develop is independent per row, so a frame
//! is split into one band of rows per core. Spawning a scoped thread costs
//! tens of microseconds, which is noise against a frame's milliseconds;
//! small images run on the calling thread.

use std::num::NonZeroUsize;
use std::sync::OnceLock;

/// Below this many values the work runs on the calling thread.
const MIN_PARALLEL: usize = 64 * 1024;
const MAX_THREADS: usize = 16;

fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, NonZeroUsize::get)
            .clamp(1, MAX_THREADS)
    })
}

/// Calls `f(y, row)` for every `row_len`-long row of `out`, across threads.
/// Rows are visited in order within a band; bands run concurrently.
pub fn rows<T, F>(out: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if row_len == 0 || out.is_empty() {
        return;
    }
    let n_rows = out.len() / row_len;
    let threads = if out.len() < MIN_PARALLEL {
        1
    } else {
        threads().min(n_rows.max(1))
    };
    if threads <= 1 {
        for (y, row) in out.chunks_mut(row_len).enumerate() {
            f(y, row);
        }
        return;
    }
    let band_rows = n_rows.div_ceil(threads);
    let f = &f;
    std::thread::scope(|s| {
        for (b, band) in out.chunks_mut(band_rows * row_len).enumerate() {
            s.spawn(move || {
                for (i, row) in band.chunks_mut(row_len).enumerate() {
                    f(b * band_rows + i, row);
                }
            });
        }
    });
}

/// `f` over consecutive bands of `data`, one per thread, returning each
/// band's result in order. For reductions (histograms, min/max) that are
/// merged by the caller.
pub fn bands<T, R, F>(data: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&[T]) -> R + Sync,
{
    let threads = if data.len() < MIN_PARALLEL {
        1
    } else {
        threads()
    };
    if threads <= 1 {
        return vec![f(data)];
    }
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(data.len().div_ceil(threads))
            .map(|band| s.spawn(move || f(band)))
            .collect();
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(r) => r,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_is_visited_once_with_its_index() {
        for (w, h) in [(3, 5), (1000, 400), (512, 513)] {
            let mut v = vec![usize::MAX; w * h];
            rows(&mut v, w, |y, row| row.iter_mut().for_each(|p| *p = y));
            for (i, p) in v.iter().enumerate() {
                assert_eq!(*p, i / w);
            }
        }
    }

    #[test]
    fn bands_cover_the_data_in_order() {
        let data: Vec<u32> = (0..200_000).collect();
        let sums = bands(&data, |b| b.iter().map(|&v| u64::from(v)).sum::<u64>());
        assert_eq!(
            sums.iter().sum::<u64>(),
            data.iter().map(|&v| u64::from(v)).sum()
        );
        let firsts = bands(&data, |b| b[0]);
        assert!(firsts.windows(2).all(|w| w[0] < w[1]));
    }
}
