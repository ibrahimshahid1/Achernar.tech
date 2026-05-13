use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Find cmake binary - cargo build scripts don't always inherit the full user PATH
fn find_cmake() -> String {
    // Check PATH first
    if Command::new("cmake").arg("--version").output().is_ok() {
        return "cmake".to_string();
    }
    // Common locations
    for path in &[
        "/opt/homebrew/bin/cmake",
        "/usr/local/bin/cmake",
        "/Applications/CMake.app/Contents/bin/cmake",
    ] {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }
    panic!("cmake not found! Install with: brew install cmake");
}

fn main() {
    let cmake = find_cmake();
    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let bridge_dir = PathBuf::from(&manifest_dir).join("ols-bridge");
    let build_dir = PathBuf::from(&out_dir).join("ols-bridge-build");

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build dir");

    // Run cmake configure
    let cmake_status = Command::new(&cmake)
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
    let build_status = Command::new(&cmake)
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

    // ─── Vendor Camera SDK linking ───
    // These SDKs are auto-detected by CMake, but Rust linker also
    // needs to know where to find the dynamic libraries.
    let project_root = PathBuf::from(&manifest_dir).parent().unwrap().to_path_buf();
    let sdks_root = project_root.join("sdks");

    // ASI (ZWO) Camera SDK
    let asi_sdk = sdks_root.join("ASI/ASI_Camera_SDK/ASI_linux_mac_SDK_V1.41");
    if asi_sdk.join("include/ASICamera2.h").exists() {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let asi_lib = asi_sdk.join("lib/mac_arm64");
        #[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
        let asi_lib = asi_sdk.join("lib/mac");
        #[cfg(target_os = "linux")]
        let asi_lib = asi_sdk.join("lib/x64");
        if asi_lib.exists() {
            #[cfg(target_os = "macos")]
            let link_dir =
                prepare_macos_dylib(&asi_lib, "ASICamera2", std::path::Path::new(&out_dir));
            #[cfg(target_os = "linux")]
            let link_dir = asi_lib.clone();

            println!("cargo:rustc-link-search=native={}", link_dir.display());
            println!("cargo:rustc-link-lib=dylib=ASICamera2");
            // Runtime rpath so dyld can find the original library in original SDK folder
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", asi_lib.display());
        }
    }

    // SVBony Camera SDK
    let svb_sdk = sdks_root.join("Svbony/macos/SVBCameraSDK");
    if svb_sdk.join("include/SVBCameraSDK.h").exists() {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let svb_lib = svb_sdk.join("lib/arm64");
        #[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
        let svb_lib = svb_sdk.join("lib/x64");
        #[cfg(target_os = "linux")]
        let svb_lib = svb_sdk.join("lib/x64");
        if svb_lib.exists() {
            #[cfg(target_os = "macos")]
            let link_dir =
                prepare_macos_dylib(&svb_lib, "SVBCameraSDK", std::path::Path::new(&out_dir));
            #[cfg(target_os = "linux")]
            let link_dir = svb_lib.clone();

            println!("cargo:rustc-link-search=native={}", link_dir.display());
            println!("cargo:rustc-link-lib=dylib=SVBCameraSDK");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", svb_lib.display());
        }
    }

    // ToupTek Camera SDK
    let toup_sdk = sdks_root.join("touptek");
    if toup_sdk.join("inc/toupcam.h").exists() {
        #[cfg(target_os = "macos")]
        let toup_lib = toup_sdk.join("mac");
        #[cfg(target_os = "linux")]
        let toup_lib = toup_sdk.join("linux/x64");
        if toup_lib.exists() {
            #[cfg(target_os = "macos")]
            let link_dir =
                prepare_macos_dylib(&toup_lib, "toupcam", std::path::Path::new(&out_dir));
            #[cfg(target_os = "linux")]
            let link_dir = toup_lib.clone();

            println!("cargo:rustc-link-search=native={}", link_dir.display());
            println!("cargo:rustc-link-lib=dylib=toupcam");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", toup_lib.display());
        }
    }

    // Rebuild if bridge sources change
    println!("cargo:rerun-if-changed=ols-bridge/src/ols_bridge.cpp");
    println!("cargo:rerun-if-changed=ols-bridge/src/camera_base.cpp");
    println!("cargo:rerun-if-changed=ols-bridge/include/ols_bridge.h");
    println!("cargo:rerun-if-changed=ols-bridge/CMakeLists.txt");
    println!("cargo:rerun-if-changed=ols-bridge/shims/cppcms/json.h");
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/aio/endpoint.h");
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/aio/basic_socket.h");
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/log.h");
}

struct PkgConfigResult {
    link_paths: Vec<PathBuf>,
    libs: Vec<String>,
}

fn find_pkg_config() -> String {
    if Command::new("pkg-config").arg("--version").output().is_ok() {
        return "pkg-config".to_string();
    }
    for path in &["/opt/homebrew/bin/pkg-config", "/usr/local/bin/pkg-config"] {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }
    panic!("pkg-config not found! Install with: brew install pkg-config");
}

fn pkg_config_libs(name: &str) -> PkgConfigResult {
    let pkg_config = find_pkg_config();
    let output = Command::new(&pkg_config)
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

/// Search a directory tree for a file whose name starts with `prefix`.
/// Returns the parent directory of the first match.
fn find_file_in_dir(dir: &std::path::Path, prefix: &str) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file_in_dir(&path, prefix) {
                return Some(found);
            }
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with(prefix) {
                return path.parent().map(|p| p.to_path_buf());
            }
        }
    }
    None
}

/// On macOS, vendors often ship dylibs with `@loader_path/libFoo.dylib` hardcoded.
/// We copy the library to OUT_DIR and patch its ID to `@rpath/libFoo.dylib` so the built executable
/// uses an rpath-based lookup, avoiding crash upon `cargo run`.
#[cfg(target_os = "macos")]
fn prepare_macos_dylib(
    sdk_lib_dir: &std::path::Path,
    lib_name: &str,
    out_dir: &std::path::Path,
) -> PathBuf {
    let dylib_name = format!("lib{}.dylib", lib_name);
    let src_path = sdk_lib_dir.join(&dylib_name);
    if src_path.exists() {
        let dest_path = out_dir.join(&dylib_name);

        // Copy the dylib
        let _ = std::fs::copy(&src_path, &dest_path);

        // Change the id to @rpath
        let new_id = format!("@rpath/{}", dylib_name);
        let _ = Command::new("install_name_tool")
            .args(["-id", &new_id, dest_path.to_str().unwrap()])
            .status();

        return out_dir.to_path_buf();
    }
    sdk_lib_dir.to_path_buf()
}
