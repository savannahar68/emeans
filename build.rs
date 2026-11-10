// Picks the matrix multiply: MKL with the `mkl` feature, Apple Accelerate with the
// `accelerate` feature, and the pure Rust `matrixmultiply` otherwise.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(emeans_blas)");
    println!("cargo:rerun-if-env-changed=MKL_LIB_DIR");
    let mkl = std::env::var_os("CARGO_FEATURE_MKL").is_some();
    let accelerate = std::env::var_os("CARGO_FEATURE_ACCELERATE").is_some();
    if mkl && accelerate {
        panic!("pick only one of the mkl and accelerate features");
    }
    if mkl {
        // Set MKL_LIB_DIR when libmkl_rt is not on the default search path.
        if let Ok(dir) = std::env::var("MKL_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
        println!("cargo:rustc-link-lib=mkl_rt");
        println!("cargo:rustc-cfg=emeans_blas");
    }
    if accelerate {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
            panic!("the accelerate feature only works on macOS");
        }
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=emeans_blas");
    }
}
