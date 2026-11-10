//! Time training on random data and print the final error.
//!
//! cargo run --release --example bench -- 200000 64 500

use std::time::Instant;

use emeans::Emeans;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("a number"))
        .collect();
    let (n, d, k) = match args[..] {
        [n, d, k] => (n, d, k),
        _ => (200_000, 64, 500),
    };
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let x: Vec<f32> = (0..n * d)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect();

    let start = Instant::now();
    let model = Emeans::new(d, k).train(&x).unwrap();
    let train = start.elapsed().as_secs_f64();
    let (dist, _) = model.assign(&x).unwrap();
    let mse = dist.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let r = model.recipe().unwrap();
    println!(
        "n={n} d={d} k={k} batch={} train={train:.3}s mse={mse:.5}",
        r.batch_size
    );
}
