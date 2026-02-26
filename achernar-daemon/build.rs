use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let bridge_dir = PathBuf::from(&manifest_dir).join("ols-bridge");
    let build_dir = PathBuf::from(&out_dir).join("ols-bridge-build");

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build dir");

    // Run cmake configure
    let cmake_status = Command::new("cmake")
        .args([
            "-S",
            bridge_dir.to_str().unwrap(),
            "-B",
            build_dir.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .status()
        .expect("Failed to run cmake configure");

    if !cmake_status.success() {
        panic!("cmake configure failed");
    }

    // Run cmake build
    let build_status = Command::new("cmake")
        .args([
            "--build",
            build_dir.to_str().unwrap(),
            "--config",
            "Release",
            "-j",
        ])
        .status()
        .expect("Failed to run cmake build");

    if !build_status.success() {
        panic!("cmake build failed");
    }

    // Tell cargo where to find the static library
    println!("cargo:rustc-link-search=native={}", build_dir.display());

    // Link the bridge static library
    println!("cargo:rustc-link-lib=static=ols_bridge");

    // Link C++ standard library
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=dylib=c++");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Link OpenCV - find via pkg-config
    let opencv_libs = pkg_config_libs("opencv4");
    for lib in &opencv_libs.link_paths {
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
    for lib in &opencv_libs.libs {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    // Link libtiff - find via pkg-config
    let tiff_libs = pkg_config_libs("libtiff-4");
    for lib in &tiff_libs.link_paths {
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
    for lib in &tiff_libs.libs {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    // Homebrew fallback search paths
    if std::path::Path::new("/opt/homebrew/lib").exists() {
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
    }

    // Rebuild if bridge sources change
    println!("cargo:rerun-if-changed=ols-bridge/src/ols_bridge.cpp");
    println!("cargo:rerun-if-changed=ols-bridge/include/ols_bridge.h");
    println!("cargo:rerun-if-changed=ols-bridge/CMakeLists.txt");
}

struct PkgConfigResult {
    link_paths: Vec<PathBuf>,
    libs: Vec<String>,
}

fn pkg_config_libs(name: &str) -> PkgConfigResult {
    let output = Command::new("pkg-config")
        .args(["--libs", name])
        .output()
        .unwrap_or_else(|_| panic!("pkg-config not found, cannot locate {}", name));

    if !output.status.success() {
        panic!(
            "pkg-config --libs {} failed: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let flags = String::from_utf8(output.stdout).unwrap();
    let mut link_paths = Vec::new();
    let mut libs = Vec::new();

    for flag in flags.split_whitespace() {
        if let Some(path) = flag.strip_prefix("-L") {
            link_paths.push(PathBuf::from(path));
        } else if let Some(lib) = flag.strip_prefix("-l") {
            libs.push(lib.to_string());
        }
    }

    PkgConfigResult { link_paths, libs }
}
