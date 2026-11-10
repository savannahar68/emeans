fn main() {
    if std::env::var_os("CARGO_FEATURE_ACCELERATE").is_some() {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
            panic!("the accelerate feature only works on macOS");
        }
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
