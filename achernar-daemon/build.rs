use std::env;
use std::path::{Path, PathBuf};
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
        r"C:\Program Files\CMake\bin\cmake.exe",
        r"C:\Program Files (x86)\CMake\bin\cmake.exe",
    ] {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }
    panic!(
        "cmake not found; install CMake 3.21+ (macOS: brew install cmake, \
         Windows: winget install Kitware.CMake)"
    );
}

fn main() {
    let cmake = find_cmake();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo target OS is available");
    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo target architecture is available");
    let bridge_dir = manifest_dir.join("ols-bridge");
    let build_dir = out_dir.join("ols-bridge-build");

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build dir");

    // Configure the native bridge. On Windows, vcpkg's CMake toolchain owns
    // OpenCV/TIFF discovery; pkg-config is intentionally not required.
    let mut configure = Command::new(&cmake);
    configure
        .arg("-S")
        .arg(&bridge_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release");

    if target_os == "windows" {
        configure.arg("-DACHERNAR_ENABLE_VENDOR_DRIVERS=OFF");

        // CMake's default Windows generator is Visual Studio, which may still
        // default to Win32 even when Rust is targeting x86_64-pc-windows-msvc.
        // `-A` is valid for Visual Studio generators but not Ninja.
        let generator = env::var("CMAKE_GENERATOR").ok();
        if generator
            .as_deref()
            .map_or(true, |name| name.contains("Visual Studio"))
        {
            configure
                .arg("-A")
                .arg(windows_cmake_platform(&target_arch));
        }

        let native_dir = out_dir.join("native");
        std::fs::create_dir_all(&native_dir).expect("Failed to create native output dir");
        for variable in [
            "CMAKE_RUNTIME_OUTPUT_DIRECTORY",
            "CMAKE_LIBRARY_OUTPUT_DIRECTORY",
            "CMAKE_ARCHIVE_OUTPUT_DIRECTORY",
            "CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE",
            "CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE",
            "CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE",
        ] {
            configure.arg(format!("-D{variable}={}", native_dir.display()));
        }

        let toolchain = env::var_os("CMAKE_TOOLCHAIN_FILE")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("VCPKG_ROOT")
                    .map(PathBuf::from)
                    .map(|root| root.join("scripts/buildsystems/vcpkg.cmake"))
            })
            .unwrap_or_else(|| {
                panic!(
                    "Windows builds require vcpkg. Set VCPKG_ROOT to the vcpkg \
                     checkout, or CMAKE_TOOLCHAIN_FILE to vcpkg.cmake"
                )
            });
        if !toolchain.is_file() {
            panic!("vcpkg CMake toolchain not found at {}", toolchain.display());
        }

        let triplet = env::var("VCPKG_TARGET_TRIPLET")
            .or_else(|_| env::var("VCPKG_DEFAULT_TRIPLET"))
            .unwrap_or_else(|_| windows_vcpkg_triplet(&target_arch).to_string());
        let installed_dir = env::var_os("VCPKG_INSTALLED_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.join("vcpkg_installed"));

        configure
            .arg(format!("-DCMAKE_TOOLCHAIN_FILE={}", toolchain.display()))
            .arg(format!("-DVCPKG_TARGET_TRIPLET={triplet}"))
            .arg(format!("-DVCPKG_MANIFEST_DIR={}", manifest_dir.display()))
            .arg(format!("-DVCPKG_INSTALLED_DIR={}", installed_dir.display()));
    }

    let cmake_status = configure.status().expect("Failed to run cmake configure");

    if !cmake_status.success() {
        panic!("cmake configure failed");
    }

    // Run cmake build
    let build_status = Command::new(&cmake)
        .arg("--build")
        .arg(&build_dir)
        .args(["--config", "Release", "--parallel"])
        .status()
        .expect("Failed to run cmake build");

    if !build_status.success() {
        panic!("cmake build failed");
    }

    // Single-config generators place outputs at the build root; Visual Studio
    // and other multi-config generators place Release outputs one level down.
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("Release").display()
    );

    if target_os == "windows" {
        // Link the import library for ols_bridge.dll. CMake owns its complete
        // OpenCV/TIFF/Media Foundation link graph.
        println!("cargo:rustc-link-lib=dylib=ols_bridge");
        println!(
            "cargo:rustc-link-search=native={}",
            out_dir.join("native").display()
        );
        stage_windows_runtime_dlls(&out_dir);
    } else {
        println!("cargo:rustc-link-lib=static=ols_bridge");
    }

    // Link C++ standard library
    match target_os.as_str() {
        "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
        "linux" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        _ => {}
    }

    if target_os != "windows" {
        link_unix_native_dependencies(&manifest_dir, &out_dir, &target_os, &target_arch);
    }

    // Rebuild if bridge sources change
    println!("cargo:rerun-if-changed=ols-bridge/src/ols_bridge.cpp");
    println!("cargo:rerun-if-changed=ols-bridge/src/camera_base.cpp");
    println!("cargo:rerun-if-changed=ols-bridge/src/windows_mf_camera.cpp");
    println!("cargo:rerun-if-changed=ols-bridge/include/ols_bridge.h");
    println!("cargo:rerun-if-changed=ols-bridge/CMakeLists.txt");
    println!("cargo:rerun-if-changed=vcpkg.json");
    println!("cargo:rerun-if-changed=ols-bridge/shims/cppcms/json.h");
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/aio/endpoint.h");
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/aio/basic_socket.h");
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/log.h");
    println!("cargo:rerun-if-changed=ols-bridge/shims/booster/regex.h");
    println!("cargo:rerun-if-env-changed=INDI_ROOT");
    println!("cargo:rerun-if-env-changed=INDI_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_GENERATOR");
    println!("cargo:rerun-if-env-changed=CMAKE_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=VCPKG_TARGET_TRIPLET");
    println!("cargo:rerun-if-env-changed=VCPKG_DEFAULT_TRIPLET");
    println!("cargo:rerun-if-env-changed=VCPKG_INSTALLED_DIR");
    for path in [
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_camera.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_client.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_mount.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/alpaca_focuser.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/mount.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/focuser.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/util.cpp",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/src/tiffmat.cpp",
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
        "../OpenLiveStacker_Drivers/OpenLiveStacker/include/os_util.h",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/include/util.h",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/include/tiffmat.h",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/include/libraw_wrapper.h",
        "../OpenLiveStacker_Drivers/OpenLiveStacker/include/sw_bin.h",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn windows_vcpkg_triplet(target_arch: &str) -> &'static str {
    match target_arch {
        "x86_64" => "x64-windows",
        "x86" => "x86-windows",
        "aarch64" => "arm64-windows",
        other => panic!("unsupported Windows target architecture: {other}"),
    }
}

fn windows_cmake_platform(target_arch: &str) -> &'static str {
    match target_arch {
        "x86_64" => "x64",
        "x86" => "Win32",
        "aarch64" => "ARM64",
        other => panic!("unsupported Windows target architecture: {other}"),
    }
}

/// Copy the bridge and its vcpkg runtime dependencies next to both normal
/// Cargo binaries and test binaries. `cargo run` augments PATH with native
/// link-search directories, but a directly launched daemon (and some test
/// runners) does not. Staging makes the resulting target directory portable
/// without relying on developer-shell environment state.
fn stage_windows_runtime_dlls(out_dir: &Path) {
    let native_dir = out_dir.join("native");
    let profile_dir = out_dir
        .parent() // .../build/achernar-daemon-<hash>
        .and_then(Path::parent) // .../build
        .and_then(Path::parent) // .../{debug,release}
        .unwrap_or_else(|| panic!("unexpected Cargo OUT_DIR layout: {}", out_dir.display()));
    let test_dir = profile_dir.join("deps");
    std::fs::create_dir_all(&test_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create Windows test runtime directory {}: {error}",
            test_dir.display()
        )
    });

    let entries = std::fs::read_dir(&native_dir).unwrap_or_else(|error| {
        panic!(
            "failed to inspect Windows native output {}: {error}",
            native_dir.display()
        )
    });
    let mut dll_count = 0;
    for entry in entries {
        let path = entry.expect("failed to read native build output").path();
        let is_dll = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map_or(false, |extension| extension.eq_ignore_ascii_case("dll"));
        if !is_dll {
            continue;
        }
        let file_name = path.file_name().expect("native DLL has a filename");
        for destination_dir in [profile_dir, test_dir.as_path()] {
            let destination = destination_dir.join(file_name);
            std::fs::copy(&path, &destination).unwrap_or_else(|error| {
                panic!(
                    "failed to stage {} at {}: {error}",
                    path.display(),
                    destination.display()
                )
            });
        }
        dll_count += 1;
    }

    if dll_count == 0 {
        panic!(
            "Windows native build produced no runtime DLLs in {}",
            native_dir.display()
        );
    }
}

/// Emit native dependency link flags for the Unix static-bridge build.
/// Windows deliberately does not enter this path: its DLL is fully linked by
/// CMake and exposes only the flat bridge ABI to Rust.
fn link_unix_native_dependencies(
    manifest_dir: &Path,
    out_dir: &Path,
    target_os: &str,
    target_arch: &str,
) {
    for package in ["opencv4", "libtiff-4"] {
        let libs = pkg_config_libs(package);
        for path in &libs.link_paths {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
        for lib in &libs.libs {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }

    if Path::new("/opt/homebrew/lib").exists() {
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
    }

    let project_root = manifest_dir
        .parent()
        .expect("daemon lives directly below the project root");
    let sdks_root = project_root.join("sdks");

    let asi_sdk = sdks_root.join("ASI/ASI_Camera_SDK/ASI_linux_mac_SDK_V1.41");
    if asi_sdk.join("include/ASICamera2.h").exists() {
        let asi_lib = match (target_os, target_arch) {
            ("macos", "aarch64") => asi_sdk.join("lib/mac_arm64"),
            ("macos", _) => asi_sdk.join("lib/mac"),
            ("linux", _) => asi_sdk.join("lib/x64"),
            _ => PathBuf::new(),
        };
        if asi_lib.is_dir() {
            let link_dir = if target_os == "macos" {
                prepare_macos_dylib(&asi_lib, "ASICamera2", out_dir)
            } else {
                asi_lib.clone()
            };
            println!("cargo:rustc-link-search=native={}", link_dir.display());
            println!("cargo:rustc-link-lib=dylib=ASICamera2");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", asi_lib.display());
        }
    }

    let svb_sdk = sdks_root.join("Svbony/macos/SVBCameraSDK");
    if svb_sdk.join("include/SVBCameraSDK.h").exists() {
        let svb_lib = match (target_os, target_arch) {
            ("macos", "aarch64") => svb_sdk.join("lib/arm64"),
            ("macos", _) | ("linux", _) => svb_sdk.join("lib/x64"),
            _ => PathBuf::new(),
        };
        if svb_lib.is_dir() {
            let link_dir = if target_os == "macos" {
                prepare_macos_dylib(&svb_lib, "SVBCameraSDK", out_dir)
            } else {
                svb_lib.clone()
            };
            println!("cargo:rustc-link-search=native={}", link_dir.display());
            println!("cargo:rustc-link-lib=dylib=SVBCameraSDK");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", svb_lib.display());
        }
    }

    let toup_sdk = sdks_root.join("touptek");
    if toup_sdk.join("inc/toupcam.h").exists() {
        let toup_lib = match target_os {
            "macos" => toup_sdk.join("mac"),
            "linux" => toup_sdk.join("linux/x64"),
            _ => PathBuf::new(),
        };
        if toup_lib.is_dir() {
            let link_dir = if target_os == "macos" {
                prepare_macos_dylib(&toup_lib, "toupcam", out_dir)
            } else {
                toup_lib.clone()
            };
            println!("cargo:rustc-link-search=native={}", link_dir.display());
            println!("cargo:rustc-link-lib=dylib=toupcam");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", toup_lib.display());
        }
    }

    link_optional_pkg("libgphoto2");
    link_optional_pkg("libuvc");
    link_optional_indi();
    link_optional_libraw();

    for include_dir in [
        "/opt/homebrew/include",
        "/usr/local/include",
        "/usr/include",
    ] {
        if Path::new(include_dir).join("indigo/indigo_bus.h").exists() {
            if let Some(prefix) = Path::new(include_dir).parent() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    prefix.join("lib").display()
                );
            }
            println!("cargo:rustc-link-lib=dylib=indigo");
            break;
        }
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

/// PKG_CONFIG_PATH augmented with Homebrew keg-only prefixes.
///
/// Homebrew's unversioned `opencv` formula now tracks OpenCV 5, which ships
/// `opencv5.pc`. OpenCV 4 is available only through the `opencv@4` formula,
/// which is keg-only — not symlinked into the brew prefix — so `opencv4.pc` is
/// invisible to pkg-config unless its directory is named explicitly.
///
/// `ols-bridge/CMakeLists.txt` performs the same probe, but that only fixes the
/// CMake process: `set(ENV{...})` there does not propagate back to this build
/// script, which runs its own `pkg-config` to emit link flags. Both layers must
/// resolve it independently, otherwise the bridge compiles and then linking
/// fails — which is exactly what happened.
///
/// Computed once; a missing formula or a missing `brew` is skipped silently so
/// Linux and non-Homebrew macOS builds are unaffected.
fn pkg_config_path() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let brew = ["brew", "/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
                .into_iter()
                .find(|candidate| {
                    Command::new(candidate)
                        .arg("--version")
                        .output()
                        .map(|out| out.status.success())
                        .unwrap_or(false)
                })?;

            let mut dirs = Vec::new();
            for formula in ["opencv@4", "libtiff", "libraw", "libgphoto2"] {
                let Ok(output) = Command::new(brew).args(["--prefix", formula]).output() else {
                    continue;
                };
                if !output.status.success() {
                    continue;
                }
                let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let pkgconfig = format!("{prefix}/lib/pkgconfig");
                if std::path::Path::new(&pkgconfig).is_dir() {
                    dirs.push(pkgconfig);
                }
            }
            // Anything the developer already exported wins as a later entry,
            // so an explicit PKG_CONFIG_PATH is respected, never discarded.
            if let Ok(existing) = env::var("PKG_CONFIG_PATH") {
                if !existing.is_empty() {
                    dirs.push(existing);
                }
            }
            (!dirs.is_empty()).then(|| dirs.join(":"))
        })
        .as_deref()
}

fn pkg_config_libs(name: &str) -> PkgConfigResult {
    let pkg_config = find_pkg_config();
    let mut command = Command::new(&pkg_config);
    command.args(["--libs", name]);
    if let Some(path) = pkg_config_path() {
        command.env("PKG_CONFIG_PATH", path);
    }
    let output = command
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
    let mut command = Command::new(&pkg_config);
    command.args(["--libs", name]);
    if let Some(path) = pkg_config_path() {
        command.env("PKG_CONFIG_PATH", path);
    }
    let output = match command.output() {
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

/// Link LibRaw, the RAW decoder the gphoto2 driver needs for CR2/CR3/RW2.
///
/// `ols-bridge/CMakeLists.txt` compiles `libraw_wrapper.h` into the *static*
/// bridge when it finds LibRaw, which leaves the actual symbols
/// (`LibRaw::open_buffer`, `LibRaw::unpack`, …) undefined until the final Rust
/// link stage — exactly like gphoto2 and libuvc above. Missing this call
/// produces a wall of undefined-symbol errors that look like a bridge bug
/// rather than a missing link flag.
///
/// pkg-config is preferred; the manual fallback covers source installs and
/// distros that ship the library without a .pc file.
fn link_optional_libraw() {
    if link_optional_pkg("libraw") || link_optional_pkg("libraw_r") {
        return;
    }

    for prefix in ["/opt/homebrew", "/usr/local", "/usr", "/opt/local"] {
        let lib_dir = PathBuf::from(prefix).join("lib");
        // Prefer the thread-safe variant when both are present; LibRaw
        // instances are constructed per-frame on the capture thread.
        for stem in ["raw_r", "raw"] {
            let found = ["dylib", "so", "a"]
                .iter()
                .any(|ext| lib_dir.join(format!("lib{stem}.{ext}")).exists());
            if found {
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                println!("cargo:rustc-link-lib=dylib={stem}");
                return;
            }
        }
    }
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

/// On macOS, vendors often ship dylibs with `@loader_path/libFoo.dylib` hardcoded.
/// We copy the library to OUT_DIR and patch its ID to `@rpath/libFoo.dylib` so the built executable
/// uses an rpath-based lookup, avoiding crash upon `cargo run`.
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
