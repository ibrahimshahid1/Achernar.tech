//! ASTAP command-line backend for [`super::plate_solver`].
//!
//! ASTAP (Han Kleijn, <https://www.hnsky.org/astap.htm>) is the solver NINA,
//! SharpCap and most capture programs call; NINA ships no solver of its own.
//! This drives it the way NINA's `ASTAPSolver` does — a private working copy of
//! the frame, explicit arguments, the result read from the `.ini` ASTAP writes
//! beside its input, the process killed if it overruns — but it is written
//! against ASTAP's documented command line, not translated from NINA's MPL-2.0
//! source.
//!
//! # Every hint is explicit
//!
//! When `-fov`, `-ra` or `-spd` is omitted, ASTAP falls back to the FITS header
//! and then to "learn mode": the field and position of the last successful
//! solve *on this computer*. That is state outside the archive — the same frame
//! could solve differently depending on what was solved before it, and nothing
//! in the record would say why. So all three are always passed: the hint when
//! there is one, otherwise a full-sky search (`-r 180`) from a fixed start, and
//! `-fov 0`, which makes ASTAP try fields from 10° down to 0.3°.
//!
//! # Reading the result
//!
//! The `.ini` says *what* was found; the exit code says *whether to trust it*.
//! A solution written by a process that then reported an error is discarded.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use super::plate_solver::{PlateSolution, SolveFailure, SolveFailureKind, SolveRequest, TanWcs};
use crate::config::PlateSolveConfig;
use crate::fits::FitsImageGeometry;

/// File stem of the working copy. ASTAP writes `<stem>.ini` (always) and
/// `<stem>.wcs` (on success) beside its input.
pub(crate) const INPUT_STEM: &str = "frame";

/// Console output kept for a failure message. ASTAP is terse; this is enough
/// for its last few lines without letting a misbehaving binary fill the log.
const CONSOLE_TAIL_CHARS: usize = 2000;

#[derive(Debug, Clone)]
pub struct AstapSolver {
    executable: PathBuf,
    database_dir: Option<PathBuf>,
    database: Option<String>,
    downsample: u32,
    max_stars: u32,
    timeout: Duration,
}

impl AstapSolver {
    pub fn new(executable: PathBuf, config: &PlateSolveConfig) -> Self {
        let database_dir = config
            .database_dir
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| databases_beside(&executable));
        Self {
            executable,
            database_dir,
            database: config.database.clone(),
            downsample: config.downsample,
            max_stars: config.max_stars,
            timeout: Duration::from_secs(config.timeout_s),
        }
    }

    pub fn executable(&self) -> &Path {
        &self.executable
    }

    /// The `-d` directory passed to ASTAP; `None` leaves the choice to ASTAP.
    pub fn database_dir(&self) -> Option<&Path> {
        self.database_dir.as_deref()
    }

    /// The command line for solving `image`. Nothing is left to ASTAP's memory
    /// of earlier solves; see the module docs.
    pub fn arguments(
        &self,
        image: &Path,
        request: &SolveRequest,
        geometry: &FitsImageGeometry,
        fov_height_deg: Option<f64>,
    ) -> Vec<OsString> {
        let mut args: Vec<OsString> = Vec::new();
        let mut push = |flag: &str, value: OsString| {
            args.push(flag.into());
            args.push(value);
        };
        push("-f", image.as_os_str().to_owned());
        // ASTAP's -fov is the image HEIGHT, not the width or the diagonal.
        push("-fov", number(fov_height_deg.unwrap_or(0.0)));
        push("-z", self.downsample.to_string().into());
        push("-s", self.max_stars.to_string().into());
        match request.hint {
            Some(hint) => {
                push("-r", number(hint.search_radius_deg));
                push("-ra", number(hint.ra_hours));
                // South-pole distance, so always positive.
                push("-spd", number(hint.dec_deg + 90.0));
            }
            None => {
                push("-r", "180".into());
                push("-ra", "0".into());
                push("-spd", "90".into());
            }
        }
        // An undebayered mosaic at full resolution: ASTAP's check-pattern
        // filter removes the CFA grid before star detection. Binned output has
        // no grid left to remove, and the option is documented for 1x1 only.
        if geometry.planes == 1
            && geometry.bayer_pattern.is_some()
            && geometry.binning.unwrap_or(1) <= 1
        {
            push("-check", "y".into());
        }
        if let Some(dir) = &self.database_dir {
            push("-d", dir.as_os_str().to_owned());
        }
        if let Some(database) = &self.database {
            push("-D", database.into());
        }
        args
    }

    /// Run ASTAP on `image`, which must sit in `work_dir`.
    ///
    /// Returns the outcome, the process wall time in milliseconds, and the
    /// command line as it should be recorded.
    pub async fn solve(
        &self,
        image: &Path,
        work_dir: &Path,
        request: &SolveRequest,
        geometry: &FitsImageGeometry,
        fov_height_deg: Option<f64>,
    ) -> (Result<PlateSolution, SolveFailure>, u64, Vec<String>) {
        let args = self.arguments(image, request, geometry, fov_height_deg);
        let recorded = self.recorded_command(&args, image);
        let started = Instant::now();
        let outcome = self.run(&args, work_dir, geometry).await;
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        (outcome, elapsed_ms, recorded)
    }

    async fn run(
        &self,
        args: &[OsString],
        work_dir: &Path,
        geometry: &FitsImageGeometry,
    ) -> Result<PlateSolution, SolveFailure> {
        let mut command = tokio::process::Command::new(&self.executable);
        command
            .args(args)
            .current_dir(work_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(windows)]
        {
            // astap.exe is a GUI program; without this it can open a window on
            // the observing computer's desktop for every solve.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let child = command.spawn().map_err(|error| {
            SolveFailure::new(
                SolveFailureKind::SolverUnavailable,
                format!("Could not start {}: {error}", self.executable.display()),
            )
        })?;
        let output = match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                return Err(SolveFailure::new(
                    SolveFailureKind::SolverError,
                    format!("Waiting for ASTAP failed: {error}"),
                ))
            }
            // Dropping the wait future drops the child, and kill_on_drop ends it.
            Err(_) => {
                return Err(SolveFailure::new(
                    SolveFailureKind::Timeout,
                    format!(
                        "ASTAP did not finish within {} s and was stopped",
                        self.timeout.as_secs()
                    ),
                ))
            }
        };

        let ini = tokio::fs::read(work_dir.join(format!("{INPUT_STEM}.ini")))
            .await
            .ok()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
        let console = console_tail(&output.stdout, &output.stderr);
        interpret(
            output.status.code(),
            ini.as_deref(),
            &console,
            [geometry.width, geometry.height],
        )
    }

    /// Executable name and arguments, with the private working copy reduced to
    /// its file name: the temporary directory is gone once the solve returns.
    fn recorded_command(&self, args: &[OsString], image: &Path) -> Vec<String> {
        let file_name = |path: &Path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string())
        };
        std::iter::once(file_name(&self.executable))
            .chain(args.iter().map(|arg| {
                if arg.as_os_str() == image.as_os_str() {
                    file_name(image)
                } else {
                    arg.to_string_lossy().into_owned()
                }
            }))
            .collect()
    }
}

/// Decide what one ASTAP run established. Pure, so every combination of exit
/// code and output file is testable without the program installed.
pub(crate) fn interpret(
    exit_code: Option<i32>,
    ini: Option<&str>,
    console: &str,
    naxis: [u32; 2],
) -> Result<PlateSolution, SolveFailure> {
    let Some(ini) = ini else {
        // ASTAP documents the .ini as always written. Without one, the exit
        // code is all there is; a clean exit with no result is not a solution.
        let kind = match exit_code {
            Some(0) => SolveFailureKind::SolverError,
            other => kind_for_exit(other),
        };
        return Err(SolveFailure::new(
            kind,
            with_console(
                format!("ASTAP wrote no result file ({})", describe_exit(exit_code)),
                console,
            ),
        ));
    };

    let fields = parse_ini(ini);
    if fields.get("PLTSOLVD").map(String::as_str) != Some("T") {
        // astap_cli writes the .ini with the specific errorlevel's ERROR line
        // and then exits 1 for every failure (astap_command_line.lpr: the
        // no-solution branch sets errorlevel:=1 after write_ini). So a specific
        // exit code wins when a build reports one, and otherwise the ERROR line
        // is the only record of a missing database versus a cloudy frame.
        let kind = match exit_code {
            Some(2 | 16 | 32 | 33) => kind_for_exit(exit_code),
            _ => fields
                .get("ERROR")
                .and_then(|error| kind_for_error(error))
                .unwrap_or(match exit_code {
                    // Reported unsolved but exited cleanly: believe "unsolved".
                    Some(0) => SolveFailureKind::NoSolution,
                    other => kind_for_exit(other),
                }),
        };
        let detail = solver_message(&fields).unwrap_or_else(|| describe_exit(exit_code));
        return Err(SolveFailure::new(kind, detail));
    }
    if exit_code != Some(0) {
        return Err(SolveFailure::new(
            SolveFailureKind::SolverError,
            with_console(
                format!(
                    "ASTAP wrote a solution and then exited with {}; the solution is not trusted",
                    describe_exit(exit_code)
                ),
                console,
            ),
        ));
    }

    let wcs = wcs_from_fields(&fields, naxis)
        .map_err(|error| SolveFailure::new(SolveFailureKind::SolverError, error))?;
    let warning = fields
        .get("WARNING")
        .map(|w| w.trim().to_string())
        .filter(|w| !w.is_empty());
    PlateSolution::from_wcs(wcs, warning).map_err(|error| {
        SolveFailure::new(
            SolveFailureKind::SolverError,
            format!("ASTAP's solution is unusable: {error}"),
        )
    })
}

/// `KEY=value` lines. ASTAP on Windows writes CRLF; `lines()` handles both.
pub(crate) fn parse_ini(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.trim().to_ascii_uppercase(), value.trim().to_string()))
        .collect()
}

fn wcs_from_fields(fields: &HashMap<String, String>, naxis: [u32; 2]) -> Result<TanWcs, String> {
    let get = |key: &str| -> Result<f64, String> {
        let raw = fields
            .get(key)
            .ok_or_else(|| format!("ASTAP's solution has no {key}"))?;
        raw.parse::<f64>()
            .map_err(|_| format!("ASTAP's {key} is not a number: {raw:?}"))
    };
    Ok(TanWcs {
        crpix: [get("CRPIX1")?, get("CRPIX2")?],
        crval: [get("CRVAL1")?, get("CRVAL2")?],
        cd: [
            [get("CD1_1")?, get("CD1_2")?],
            [get("CD2_1")?, get("CD2_2")?],
        ],
        naxis,
    })
}

/// ASTAP's documented errorlevels.
fn kind_for_exit(code: Option<i32>) -> SolveFailureKind {
    match code {
        Some(1) => SolveFailureKind::NoSolution,
        Some(2) => SolveFailureKind::NotEnoughStars,
        Some(16) => SolveFailureKind::ImageUnreadable,
        Some(32) => SolveFailureKind::StarDatabaseMissing,
        Some(33) => SolveFailureKind::StarDatabaseUnreadable,
        _ => SolveFailureKind::SolverError,
    }
}

/// The fixed `ERROR=` texts ASTAP's `write_ini` emits for its errorlevels.
fn kind_for_error(error: &str) -> Option<SolveFailureKind> {
    let error = error.trim().trim_end_matches('.').to_ascii_lowercase();
    Some(match error.as_str() {
        "not enough stars" => SolveFailureKind::NotEnoughStars,
        "error reading image file" => SolveFailureKind::ImageUnreadable,
        "no star database found" => SolveFailureKind::StarDatabaseMissing,
        "error reading star database" => SolveFailureKind::StarDatabaseUnreadable,
        _ => return None,
    })
}

fn describe_exit(code: Option<i32>) -> String {
    match code {
        Some(0) => "exit code 0".into(),
        Some(1) => "exit code 1: no solution".into(),
        Some(2) => "exit code 2: not enough stars detected".into(),
        Some(16) => "exit code 16: error reading image file".into(),
        Some(32) => "exit code 32: no star database found".into(),
        Some(33) => "exit code 33: error reading star database".into(),
        Some(34) => "exit code 34: error updating input file".into(),
        Some(other) => format!("exit code {other}"),
        None => "terminated by a signal".into(),
    }
}

/// ASTAP's own explanation: ERROR, then WARNING, when either is non-empty.
fn solver_message(fields: &HashMap<String, String>) -> Option<String> {
    let parts: Vec<&str> = ["ERROR", "WARNING"]
        .iter()
        .filter_map(|key| fields.get(*key).map(|v| v.trim()))
        .filter(|v| !v.is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join("; "))
}

fn console_tail(stdout: &[u8], stderr: &[u8]) -> String {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(stdout),
        String::from_utf8_lossy(stderr)
    );
    let text = text.trim();
    let skip = text.chars().count().saturating_sub(CONSOLE_TAIL_CHARS);
    text.chars().skip(skip).collect()
}

fn with_console(message: String, console: &str) -> String {
    if console.is_empty() {
        message
    } else {
        format!("{message}. Output: {console}")
    }
}

/// Fixed-point, '.' decimal separator regardless of locale, trailing zeros
/// trimmed so the recorded command line stays readable.
fn number(value: f64) -> OsString {
    let text = format!("{value:.6}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    OsString::from(if text.is_empty() || text == "-" || text == "-0" {
        "0"
    } else {
        text
    })
}

// ═══════════════════════════════════════════════════════════════════
//  Discovery
// ═══════════════════════════════════════════════════════════════════

/// Look for ASTAP where its installers put it, then on `PATH`.
///
/// The command-line build is preferred everywhere: the GUI binary accepts the
/// same arguments but links GUI libraries that may be missing on a headless
/// observatory computer.
pub fn discover() -> Option<PathBuf> {
    for name in ["astap_cli", "astap"] {
        let on_path: Vec<PathBuf> = std::env::var_os("PATH")
            .map(|path| {
                std::env::split_paths(&path)
                    .map(|dir| dir.join(executable_name(name)))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(found) = install_locations(name)
            .into_iter()
            .chain(on_path)
            .find(|path| path.is_file())
        {
            return Some(found);
        }
    }
    None
}

/// The star database directory implied by where the executable lives.
///
/// ASTAP looks for its databases beside its own executable on Linux and
/// Windows, but at a fixed `/usr/local/opt/astap` on macOS — a directory only
/// an administrator can create. When nothing is configured and database tiles
/// sit beside the real executable (symlinks resolved), name that directory, so
/// the unzip-everything-into-one-folder install works on every platform. Every
/// ASTAP database's first tile is named `<db>_0101.<ext>`.
fn databases_beside(executable: &Path) -> Option<PathBuf> {
    let directory = std::fs::canonicalize(executable)
        .ok()?
        .parent()?
        .to_path_buf();
    let has_tiles = std::fs::read_dir(&directory)
        .ok()?
        .filter_map(Result::ok)
        .any(|entry| entry.file_name().to_string_lossy().contains("_0101."));
    has_tiles.then_some(directory)
}

/// The fixed install locations [`discover`] checks before `PATH`.
pub fn searched_locations() -> Vec<PathBuf> {
    ["astap_cli", "astap"]
        .iter()
        .flat_map(|name| install_locations(name))
        .collect()
}

fn executable_name(stem: &str) -> String {
    format!("{stem}{}", std::env::consts::EXE_SUFFIX)
}

#[cfg(target_os = "windows")]
fn install_locations(name: &str) -> Vec<PathBuf> {
    let program_files = std::env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
    vec![program_files.join("astap").join(executable_name(name))]
}

#[cfg(target_os = "macos")]
fn install_locations(name: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = [
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/usr/local/opt/astap",
    ]
    .iter()
    .map(|dir| Path::new(dir).join(name))
    .collect();
    if name == "astap" {
        paths.push(PathBuf::from(
            "/Applications/ASTAP.app/Contents/MacOS/astap",
        ));
    }
    paths.extend(user_bin(name));
    paths
}

/// `~/.local/bin`, the per-user executable directory. Listed explicitly
/// because a daemon started from a desktop launcher does not inherit the
/// login shell's PATH, so a PATH-only search would miss it.
#[cfg(unix)]
fn user_bin(name: &str) -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| Path::new(&home).join(".local/bin").join(name))
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn install_locations(name: &str) -> Vec<PathBuf> {
    ["/opt/astap", "/usr/local/bin", "/usr/bin"]
        .iter()
        .map(|dir| Path::new(dir).join(name))
        .chain(user_bin(name))
        .collect()
}

/// Stand-ins for ASTAP, shared with the API tests.
#[cfg(test)]
pub(crate) mod test_support {
    /// 1.2"/px, north up, east left, 4000 x 3000, centred on M42. Synthetic,
    /// written in ASTAP's documented `.ini` layout (CRLF, as on Windows).
    pub(crate) const SOLVED_INI: &str = "PLTSOLVD=T\r\n\
        CRPIX1= 2.0005000000000000E+003\r\n\
        CRPIX2= 1.5005000000000000E+003\r\n\
        CRVAL1= 8.3822000000000000E+001\r\n\
        CRVAL2=-5.3910000000000000E+000\r\n\
        CDELT1=-3.3333333333333332E-004\r\n\
        CDELT2= 3.3333333333333332E-004\r\n\
        CROTA1= 0.0000000000000000E+000\r\n\
        CROTA2= 0.0000000000000000E+000\r\n\
        CD1_1=-3.3333333333333332E-004\r\n\
        CD1_2= 0.0000000000000000E+000\r\n\
        CD2_1= 0.0000000000000000E+000\r\n\
        CD2_2= 3.3333333333333332E-004\r\n\
        CMDLINE=astap_cli -f frame.fits\r\n\
        WARNING=\r\n";

    /// A shell script that behaves like ASTAP: it records its arguments in
    /// `args.txt` beside itself, waits `sleep_s`, writes `ini` (if any) as
    /// `<stem>.ini` beside the `-f` image, and exits with `code`.
    #[cfg(unix)]
    pub(crate) fn fake_astap(
        dir: &std::path::Path,
        ini: Option<&str>,
        code: i32,
        sleep_s: f64,
    ) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join("astap_cli");
        let write_ini = match ini {
            Some(text) => format!(
                "cat > \"${{image%.*}}.ini\" <<'EOF'\n{}\nEOF\n",
                text.replace("\r\n", "\n")
            ),
            None => String::new(),
        };
        let body = format!(
            "#!/bin/sh\n\
             printf '%s\\n' \"$@\" > \"$(dirname \"$0\")/args.txt\"\n\
             while [ $# -gt 0 ]; do\n  \
               if [ \"$1\" = \"-f\" ]; then image=\"$2\"; fi\n  \
               shift\n\
             done\n\
             sleep {sleep_s}\n\
             {write_ini}\
             exit {code}\n"
        );
        std::fs::write(&script, body).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::SOLVED_INI;
    use super::*;
    use crate::processing::plate_solver::PointingHint;

    const NAXIS: [u32; 2] = [4000, 3000];

    fn config() -> PlateSolveConfig {
        PlateSolveConfig {
            database_dir: Some("/data/astap".into()),
            database: Some("d50".into()),
            ..PlateSolveConfig::default()
        }
    }

    fn mono() -> FitsImageGeometry {
        FitsImageGeometry {
            width: 4000,
            height: 3000,
            planes: 1,
            bitpix: 16,
            bayer_pattern: None,
            binning: Some(1),
        }
    }

    fn strings(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    fn value_after(args: &[String], flag: &str) -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    }

    #[test]
    fn a_hinted_solve_passes_position_radius_and_scale() {
        let solver = AstapSolver::new("astap_cli".into(), &config());
        let request = SolveRequest {
            hint: Some(PointingHint {
                ra_hours: 5.588,
                dec_deg: -5.391,
                search_radius_deg: 10.0,
            }),
            pixel_scale_arcsec: Some(1.2),
        };
        let args =
            strings(&solver.arguments(Path::new("/w/frame.fits"), &request, &mono(), Some(1.0)));
        assert_eq!(value_after(&args, "-f").as_deref(), Some("/w/frame.fits"));
        assert_eq!(value_after(&args, "-fov").as_deref(), Some("1"));
        assert_eq!(value_after(&args, "-r").as_deref(), Some("10"));
        assert_eq!(value_after(&args, "-ra").as_deref(), Some("5.588"));
        assert_eq!(value_after(&args, "-spd").as_deref(), Some("84.609"));
        assert_eq!(value_after(&args, "-z").as_deref(), Some("0"));
        assert_eq!(value_after(&args, "-s").as_deref(), Some("500"));
        assert_eq!(value_after(&args, "-d").as_deref(), Some("/data/astap"));
        assert_eq!(value_after(&args, "-D").as_deref(), Some("d50"));
        assert!(!args.contains(&"-check".to_string()));
    }

    #[test]
    fn a_blind_solve_still_states_every_hint() {
        // Omitting -ra/-spd/-fov would hand the search start and scale to
        // ASTAP's memory of the previous solve on this computer.
        let solver = AstapSolver::new("astap_cli".into(), &PlateSolveConfig::default());
        let args = strings(&solver.arguments(
            Path::new("frame.fits"),
            &SolveRequest::default(),
            &mono(),
            None,
        ));
        assert_eq!(value_after(&args, "-r").as_deref(), Some("180"));
        assert_eq!(value_after(&args, "-ra").as_deref(), Some("0"));
        assert_eq!(value_after(&args, "-spd").as_deref(), Some("90"));
        assert_eq!(value_after(&args, "-fov").as_deref(), Some("0"));
        assert!(!args.contains(&"-d".to_string()) && !args.contains(&"-D".to_string()));
    }

    #[test]
    fn only_a_full_resolution_mosaic_gets_the_check_pattern_filter() {
        let solver = AstapSolver::new("astap_cli".into(), &PlateSolveConfig::default());
        let request = SolveRequest::default();
        let with = |geometry: FitsImageGeometry| {
            strings(&solver.arguments(Path::new("frame.fits"), &request, &geometry, None))
                .contains(&"-check".to_string())
        };
        let mosaic = FitsImageGeometry {
            bayer_pattern: Some("RGGB".into()),
            ..mono()
        };
        assert!(with(mosaic.clone()));
        assert!(!with(FitsImageGeometry {
            binning: Some(2),
            ..mosaic.clone()
        }));
        assert!(!with(FitsImageGeometry {
            planes: 3,
            ..mosaic
        }));
        assert!(!with(mono()));
    }

    #[test]
    fn numbers_use_a_dot_and_no_exponent() {
        assert_eq!(number(30.0), "30");
        assert_eq!(number(100.0), "100");
        assert_eq!(number(84.609), "84.609");
        assert_eq!(number(0.0000001), "0");
        assert_eq!(number(1.2345678), "1.234568");
    }

    #[test]
    fn a_solved_ini_becomes_a_solution() {
        let solution = interpret(Some(0), Some(SOLVED_INI), "", NAXIS).unwrap();
        assert!((solution.ra_deg - 83.822).abs() < 1e-9);
        assert!((solution.dec_deg + 5.391).abs() < 1e-9);
        assert!((solution.pixel_scale_arcsec - 1.2).abs() < 1e-9);
        assert!(solution.position_angle_deg.abs() < 1e-9);
        assert!(!solution.mirrored);
        assert_eq!(solution.warning, None);
    }

    #[test]
    fn a_warning_on_a_solution_is_kept() {
        let ini = SOLVED_INI.replace("WARNING=", "WARNING=Hot pixels detected");
        let solution = interpret(Some(0), Some(&ini), "", NAXIS).unwrap();
        assert_eq!(solution.warning.as_deref(), Some("Hot pixels detected"));
    }

    #[test]
    fn exit_codes_name_the_failure() {
        let unsolved = "PLTSOLVD=F\nCMDLINE=astap_cli\nERROR=\nWARNING=\n";
        for (code, kind) in [
            (Some(1), SolveFailureKind::NoSolution),
            (Some(2), SolveFailureKind::NotEnoughStars),
            (Some(16), SolveFailureKind::ImageUnreadable),
            (Some(32), SolveFailureKind::StarDatabaseMissing),
            (Some(33), SolveFailureKind::StarDatabaseUnreadable),
            (Some(0), SolveFailureKind::NoSolution),
            (Some(7), SolveFailureKind::SolverError),
            (None, SolveFailureKind::SolverError),
        ] {
            let failure = interpret(code, Some(unsolved), "", NAXIS).unwrap_err();
            assert_eq!(failure.kind, kind, "exit {code:?}");
        }
    }

    #[test]
    fn astap_cli_exits_1_for_everything_so_its_error_line_names_the_failure() {
        // The ERROR texts astap_cli's write_ini emits; the exit code is 1 for
        // all of them. A missing database must not be filed as a cloudy frame.
        for (error, kind) in [
            ("Not enough stars.", SolveFailureKind::NotEnoughStars),
            (
                "Error reading image file.",
                SolveFailureKind::ImageUnreadable,
            ),
            (
                "No star database found.",
                SolveFailureKind::StarDatabaseMissing,
            ),
            (
                "Error reading star database.",
                SolveFailureKind::StarDatabaseUnreadable,
            ),
            ("Something new.", SolveFailureKind::NoSolution),
        ] {
            let ini = format!("\nPLTSOLVD=F\nCMDLINE=astap_cli -f frame.fits\nERROR={error}\n");
            let failure = interpret(Some(1), Some(&ini), "", NAXIS).unwrap_err();
            assert_eq!(failure.kind, kind, "{error}");
            assert!(failure.detail.contains(error.trim_end_matches('.')));
        }
        let plain = "\nPLTSOLVD=F\nCMDLINE=astap_cli -f frame.fits\nWARNING=Wide field image, use W08 database! \n";
        let failure = interpret(Some(1), Some(plain), "", NAXIS).unwrap_err();
        assert_eq!(failure.kind, SolveFailureKind::NoSolution);
        assert_eq!(failure.detail, "Wide field image, use W08 database!");
    }

    #[test]
    fn astaps_own_error_text_is_the_detail() {
        let ini = "PLTSOLVD=F\nERROR=No star database found at /opt/astap\nWARNING=\n";
        let failure = interpret(Some(32), Some(ini), "", NAXIS).unwrap_err();
        assert_eq!(failure.kind, SolveFailureKind::StarDatabaseMissing);
        assert_eq!(failure.detail, "No star database found at /opt/astap");
    }

    #[test]
    fn a_solution_from_a_failing_process_is_not_trusted() {
        let failure = interpret(Some(34), Some(SOLVED_INI), "", NAXIS).unwrap_err();
        assert_eq!(failure.kind, SolveFailureKind::SolverError);
        assert!(failure.detail.contains("not trusted"));
    }

    #[test]
    fn a_clean_exit_without_a_result_file_is_not_a_solution() {
        let failure = interpret(Some(0), None, "segfault-ish noise", NAXIS).unwrap_err();
        assert_eq!(failure.kind, SolveFailureKind::SolverError);
        assert!(failure.detail.contains("segfault-ish noise"));
        let failure = interpret(Some(2), None, "", NAXIS).unwrap_err();
        assert_eq!(failure.kind, SolveFailureKind::NotEnoughStars);
    }

    #[test]
    fn a_solution_missing_its_matrix_is_refused() {
        let ini = SOLVED_INI.replace("CD2_2= 3.3333333333333332E-004\r\n", "");
        let failure = interpret(Some(0), Some(&ini), "", NAXIS).unwrap_err();
        assert_eq!(failure.kind, SolveFailureKind::SolverError);
        assert!(failure.detail.contains("CD2_2"));
    }

    #[test]
    fn the_recorded_command_names_the_input_by_file_name_only() {
        let solver = AstapSolver::new("/opt/astap/astap_cli".into(), &PlateSolveConfig::default());
        let image = Path::new("/tmp/achernar-platesolve/1234/frame.fits");
        let args = solver.arguments(image, &SolveRequest::default(), &mono(), None);
        let recorded = solver.recorded_command(&args, image);
        assert_eq!(recorded[0], "astap_cli");
        assert_eq!(recorded[1..3], ["-f".to_string(), "frame.fits".to_string()]);
        assert!(!recorded.iter().any(|a| a.contains("achernar-platesolve")));
    }

    #[cfg(unix)]
    mod process {
        //! The real process path, against stand-in shell scripts that behave
        //! like ASTAP: they write `<stem>.ini` beside the `-f` argument.

        use super::super::test_support::fake_astap;
        use super::*;
        use crate::processing::plate_solver::{PlateSolver, SolveError};
        use sha2::{Digest, Sha256};

        struct Scratch(PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        fn scratch() -> Scratch {
            let dir =
                std::env::temp_dir().join(format!("achernar-astap-test-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }

        /// A small FITS on disk and its SHA-256.
        fn archived_frame(dir: &Path) -> (PathBuf, String) {
            let bytes =
                crate::fits::encode_linear_analysis(&[1.0; 48], 8, 6, "ADU", "test").unwrap();
            let path = dir.join("image.fits");
            std::fs::write(&path, &bytes).unwrap();
            (path, format!("{:x}", Sha256::digest(&bytes)))
        }

        fn solver(executable: PathBuf, work_root: PathBuf, timeout_s: u64) -> PlateSolver {
            let config = PlateSolveConfig {
                timeout_s,
                ..PlateSolveConfig::default()
            };
            PlateSolver::with_backend_for_test(AstapSolver::new(executable, &config), work_root)
        }

        #[tokio::test]
        async fn solves_a_copy_and_leaves_the_archive_untouched() {
            let dir = scratch();
            let ini = SOLVED_INI.replace("\r\n", "\n");
            let astap = fake_astap(&dir.0, Some(&ini), 0, 0.0);
            let (frame, sha) = archived_frame(&dir.0);
            let work_root = dir.0.join("work");
            let solver = solver(astap, work_root.clone(), 30);

            let attempt = solver
                .solve_archived_fits(&frame, Some(&sha), &SolveRequest::default())
                .await
                .unwrap();
            let solution = attempt.outcome.expect("fake ASTAP reported a solution");
            assert!((solution.pixel_scale_arcsec - 1.2).abs() < 1e-9);
            assert_eq!((attempt.geometry.width, attempt.geometry.height), (8, 6));
            assert_eq!(attempt.command[2], "frame.fits");

            // The solver was pointed at the working copy, never the archive,
            // and nothing was written beside the archived frame.
            let args = std::fs::read_to_string(dir.0.join("args.txt")).unwrap();
            assert!(args.contains(&work_root.display().to_string()));
            assert!(!dir.0.join("image.ini").exists());
            // The working directory went with the solve.
            assert_eq!(std::fs::read_dir(&work_root).unwrap().count(), 0);
        }

        #[tokio::test]
        async fn a_frame_that_changed_since_it_was_archived_is_not_solved() {
            let dir = scratch();
            let astap = fake_astap(&dir.0, Some("PLTSOLVD=T"), 0, 0.0);
            let (frame, _) = archived_frame(&dir.0);
            let solver = solver(astap, dir.0.join("work"), 30);
            let error = solver
                .solve_archived_fits(&frame, Some(&"0".repeat(64)), &SolveRequest::default())
                .await
                .unwrap_err();
            assert!(matches!(error, SolveError::ChecksumMismatch { .. }));
            assert!(!dir.0.join("args.txt").exists(), "solver must not have run");
        }

        #[tokio::test]
        async fn an_unsolved_frame_is_a_recorded_outcome_not_an_error() {
            let dir = scratch();
            let ini = "PLTSOLVD=F\nERROR=No solution found\nWARNING=";
            let astap = fake_astap(&dir.0, Some(ini), 1, 0.0);
            let (frame, sha) = archived_frame(&dir.0);
            let solver = solver(astap, dir.0.join("work"), 30);
            let attempt = solver
                .solve_archived_fits(&frame, Some(&sha), &SolveRequest::default())
                .await
                .unwrap();
            let failure = attempt.outcome.unwrap_err();
            assert_eq!(failure.kind, SolveFailureKind::NoSolution);
            assert_eq!(failure.detail, "No solution found");
        }

        #[tokio::test]
        async fn an_overrunning_solver_is_stopped() {
            let dir = scratch();
            let astap = fake_astap(&dir.0, Some("PLTSOLVD=T"), 0, 30.0);
            let (frame, sha) = archived_frame(&dir.0);
            let solver = solver(astap, dir.0.join("work"), 1);
            let started = Instant::now();
            let attempt = solver
                .solve_archived_fits(&frame, Some(&sha), &SolveRequest::default())
                .await
                .unwrap();
            assert_eq!(attempt.outcome.unwrap_err().kind, SolveFailureKind::Timeout);
            assert!(started.elapsed() < Duration::from_secs(10));
        }

        #[tokio::test]
        async fn a_second_solve_while_one_runs_is_refused() {
            let dir = scratch();
            let astap = fake_astap(&dir.0, Some(&SOLVED_INI.replace("\r\n", "\n")), 0, 1.0);
            let (frame, sha) = archived_frame(&dir.0);
            let solver = std::sync::Arc::new(solver(astap, dir.0.join("work"), 30));

            let first = {
                let solver = solver.clone();
                let (frame, sha) = (frame.clone(), sha.clone());
                tokio::spawn(async move {
                    solver
                        .solve_archived_fits(&frame, Some(&sha), &SolveRequest::default())
                        .await
                })
            };
            while !solver.status().busy {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let second = solver
                .solve_archived_fits(&frame, Some(&sha), &SolveRequest::default())
                .await;
            assert!(matches!(second, Err(SolveError::Busy)));
            assert!(first.await.unwrap().unwrap().outcome.is_ok());
        }

        #[test]
        fn databases_beside_the_real_executable_are_used_unless_configured() {
            let dir = scratch();
            let install = dir.0.join("astap");
            std::fs::create_dir_all(&install).unwrap();
            let executable = fake_astap(&install, None, 0, 0.0);
            let linked = dir.0.join("astap_cli");
            std::os::unix::fs::symlink(&executable, &linked).unwrap();

            // No tiles yet: nothing to infer, ASTAP keeps its own default.
            let solver = AstapSolver::new(linked.clone(), &PlateSolveConfig::default());
            assert_eq!(solver.database_dir(), None);

            // A tile beside the real binary, reached through a symlink.
            std::fs::write(install.join("d05_0101.1476"), b"").unwrap();
            let solver = AstapSolver::new(linked.clone(), &PlateSolveConfig::default());
            let expected = std::fs::canonicalize(&install).unwrap();
            assert_eq!(solver.database_dir(), Some(expected.as_path()));

            // An explicit setting always wins.
            let configured = PlateSolveConfig {
                database_dir: Some("/data/astap".into()),
                ..PlateSolveConfig::default()
            };
            let solver = AstapSolver::new(linked, &configured);
            assert_eq!(solver.database_dir(), Some(Path::new("/data/astap")));
        }

        #[tokio::test]
        async fn a_missing_executable_is_reported_not_hidden() {
            let dir = scratch();
            let (frame, sha) = archived_frame(&dir.0);
            let solver = solver(dir.0.join("no-such-astap"), dir.0.join("work"), 30);
            let attempt = solver
                .solve_archived_fits(&frame, Some(&sha), &SolveRequest::default())
                .await
                .unwrap();
            assert_eq!(
                attempt.outcome.unwrap_err().kind,
                SolveFailureKind::SolverUnavailable
            );
        }
    }
}
