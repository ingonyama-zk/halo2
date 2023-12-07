# halo2 [![Crates.io](https://img.shields.io/crates/v/halo2.svg)](https://crates.io/crates/halo2) #

## [Documentation](https://docs.rs/halo2)

## Minimum Supported Rust Version

Requires Rust **1.69.0** or higher.

Minimum supported Rust version can be changed in the future, but it will be done with a
minor version bump.

## Controlling parallelism

`halo2` currently uses [rayon](https://github.com/rayon-rs/rayon) for parallel computation.
The `RAYON_NUM_THREADS` environment variable can be used to set the number of threads.

You can disable `rayon` by disabling the `"multicore"` feature.
Warning! Halo2 will lose access to parallelism if you disable the `"multicore"` feature.
This will significantly degrade performance.

## GPU Acceleration

If you have access to NVIDIA GPUs, you can enable acceleration by building with the feature `icicle_gpu` and setting the following environment variable:

```sh
export ENABLE_ICICLE_GPU=true
```

GPU acceleration is provided by [Icicle](https://github.com/ingonyama-zk/icicle)

To go back to running with CPU, the previous environment variable must be **unset** instead of being switched to a value of false:

```sh
unset ENABLE_ICICLE_GPU
```

>**NOTE:** Even with the above environment variable set, for circuits where k <= 8, icicle is only enabled in certain areas where batching MSMs will help; all other places will fallback to using CPU MSM. To change the value of `k` where icicle is enabled, you can set the environment variable `ICICLE_IS_SMALL_CIRCUIT`.
> 
> Example: The following will cause icicle single MSM to be used throughout when k > 10 and CPU single MSM with certain locations using icicle batched MSM when k <= 10
>```sh
>export ICICLE_IS_SMALL_CIRCUIT=10
>```
>

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
