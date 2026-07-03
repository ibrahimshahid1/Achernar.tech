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

    // ─── Optional system camera libraries (gphoto2, libuvc, INDI, INDIGO) ───
    // Mirrors the CMake driver detection. The bridge is a *static* lib, so these
    // dependency libs must be linked at the final binary stage. Emit link flags
    // only when present, so the default build (none installed) is unaffected.
    link_optional_pkg("libgphoto2");
    link_optional_pkg("libuvc");
    link_optional_indi();

    // INDIGO library
    for inc in ["/opt/homebrew/include", "/usr/local/include", "/usr/include"] {
        if std::path::Path::new(&format!("{inc}/indigo/indigo_bus.h")).exists() {
            println!("cargo:rustc-link-search=native={}", inc.replace("/include", "/lib"));
            println!("cargo:rustc-link-lib=dylib=indigo");
            break;
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
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/regex.h");
    println!("cargo:rerun-if-env-changed=INDI_ROOT");
    println!("cargo:rerun-if-env-changed=INDI_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    for path in [
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_client.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_mount.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_focuser.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/asi_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/svb_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/toup_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/indi_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/indi_mount.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/gphoto2_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/wdir_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/uvc_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/indigo_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/include/camera.h",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/include/alpaca_client.h",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
}

struct PkgConfigResult {
    link_paths: Vec<PathBuf>,
    libs: Vec<String>,
}

fn find_pkg_config() -> String {
    find_pkg_config_optional().unwrap_or_else(|| {
        panic!("pkg-config not found! Install with: brew install pkg-config");
    })
}

fn find_pkg_config_optional() -> Option<String> {
    if Command::new("pkg-config").arg("--version").output().is_ok() {
        return Some("pkg-config".to_string());
    }
    for path in &["/opt/homebrew/bin/pkg-config", "/usr/local/bin/pkg-config"] {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
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

/// Emit cargo link flags for an *optional* pkg-config package. No-op (and no
/// panic) if pkg-config or the package is missing, so optional drivers don't
/// break builds on machines without their system libraries.
fn link_optional_pkg(name: &str) -> bool {
    let Some(pkg_config) = find_pkg_config_optional() else {
        return false;
    };
    let output = match Command::new(&pkg_config).args(["--libs", name]).output() {
        Ok(o) if o.status.success() => o,
        _ => return false,
    };
    let flags = String::from_utf8_lossy(&output.stdout);
    for flag in flags.split_whitespace() {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
    true
}

fn link_optional_indi() {
    if link_optional_pkg("libindi") || link_optional_pkg("indiclient") {
        return;
    }

    let mut prefixes: Vec<PathBuf> = ["INDI_ROOT", "INDI_DIR"]
        .iter()
        .filter_map(|name| env::var_os(name))
        .map(PathBuf::from)
        .collect();
    prefixes.extend(
        ["/opt/homebrew", "/usr/local", "/usr", "/opt/local"]
            .iter()
            .map(PathBuf::from),
    );

    for prefix in prefixes {
        let lib_dir = prefix.join("lib");
        let dylib = lib_dir.join("libindiclient.dylib");
        let so = lib_dir.join("libindiclient.so");
        let archive = lib_dir.join("libindiclient.a");

        let link_kind = if dylib.exists() || so.exists() {
            Some("dylib")
        } else if archive.exists() {
            Some("static")
        } else {
            None
        };

        if let Some(kind) = link_kind {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib={kind}=indiclient");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
            break;
        }
    }
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
