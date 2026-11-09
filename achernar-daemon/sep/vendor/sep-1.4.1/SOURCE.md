# Vendored SEP 1.4.1 C library

Upstream: <https://github.com/sep-developers/sep>, tag `v1.4.1`.

Source archives the files were taken from (both checked):

| Archive | sha256 |
| --- | --- |
| GitHub tag tarball `v1.4.1` (`sep-v1.4.1-github.tar.gz`) | `33ab16a55358af8c0a8b2e75e1f9f170de0037b7d8b38c660994400c13282ce1` |
| PyPI sdist `sep-1.4.1.tar.gz` | `a0c8324ab66ee716080472cb212e4303e6c3e33b0d43763075b20d4cab793b25` |

`src/` and `licenses/` are byte-identical between the two archives (the sdist
adds only `src/sep.egg-info/`).

Vendored: the C library sources and headers in `src/` (all 13 files: the
eight translation units listed in upstream `CMakeLists.txt`/`Makefile`, the
headers, and `aperture.i`, which `aperture.c` includes as a template) and
`licenses/`. Not vendored: the Python/Cython wrapper `sep.pyx` (MIT), tests,
docs, data and build scripts.

## Modifications

All files are unmodified upstream copies except `src/extract.c`, which carries
one memory-only patch, kept as
`patches/0001-extract-convert_to_catalog-remove-duplicate-allocations.patch`
(apply with `patch -p1` in this directory to an upstream copy):

- `convert_to_catalog()` allocated `cat->cflux` and `cat->flux` twice
  (`QMALLOC` after `cat->cxy` and again after `cat->ypeak`), leaking the first
  pair, `2 * sizeof(float) * nobj` bytes, on every successful `sep_extract`.
  Confirmed with macOS `leaks --atExit` on this crate's tests before the patch
  (402 leaks = 201 extractions x 2 allocations, all rooted in
  `convert_to_catalog`) and 0 leaks after it. The patch deletes the second
  pair of `QMALLOC` lines.
- No numerical effect: both arrays are written only after all allocations,
  from the same object fields. The PyPI wheel still has the leak; the parity
  fixture (`tests/fixtures/parity_v1`) is reproduced bit for bit with the
  patched file.

| File | upstream sha256 | vendored sha256 |
| --- | --- | --- |
| `src/extract.c` | `4cdd8056988bd4ff1ae63c43fef2adbfc27a839960c5d896138b416b437fbebd` | `dc28f2c79aff6fb9e7d3b64fffdd7117411fd2994d5a45a4af5cbc770119175b` |

## Per-file sha256 (as vendored)

```
5246c7d40526642b83d7b5abc949a7477592783c9e8354cf3c16ccfdaa7c6b05  src/analyse.c
6d1737ed51347d6cd81d6f85b80669f9a3729e1e4e3ff34ed4f39e957d6efc80  src/aperture.c
364d3b687ff5406ecdb023739a090efbdcaa1fc5411b7a6225682ce02b605d48  src/aperture.i
441335da11fd3d7d0a9b7c22af45630c6750cfeeec63ecf038af02bcba36bf6a  src/background.c
7d9f3b7d15721031ba4fa6a010fdec8cee25fba483a65151931d860e2b499ec9  src/convolve.c
394109e35539ed02fd02dfd56e0fa8ccf1724af8ac3d48c1df0bc5401ecf1dd0  src/deblend.c
dc28f2c79aff6fb9e7d3b64fffdd7117411fd2994d5a45a4af5cbc770119175b  src/extract.c   (patched; upstream 4cdd8056...fbebd)
ff0121b0b5510887b225e1237cfaca56475daa7af345b2c5c778a4d533ffa1b4  src/extract.h
f9841b58001449c0c66d09a84559894446fff3688bb87b9de8b75afa6a313c0f  src/lutz.c
26fcf728b70089ffc11541f27d81aa0a32badf24268d3fc1e1023dffda2f09f0  src/overlap.h
beb4465abdcbdbdaac942c280ce2128f5a800c10040a3ba885d35121bcf447e3  src/sep.h
f32a5c334847299f1bb09849eceec09c682b00d145855825d07ebc4be04bda64  src/sepcore.h
8828a4abe04205c817fef68883470a689419c6971a85d2b98118f79e4a799560  src/util.c
17b710487bfd1b940b1bd0eeb8c6924b25d5a4226efeae790beaf4478de6360a  licenses/BSD_LICENSE.txt
da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768  licenses/LGPL_LICENSE.txt
e4db5d0b438516125c511f97a8af9136a1a02478321c4b4717a5ea590a6d320d  licenses/MIT_LICENSE.txt
bdfea1646116b85d161f94a62b745be1375d2c751699d849f8293883fc88c120  patches/0001-extract-convert_to_catalog-remove-duplicate-allocations.patch
```

Check with `shasum -a 256 -c` after stripping the annotation, or compare with
the upstream archive using `diff -r`.

## Licenses

Per the file headers and upstream `README.md` ("The license for the library
as a whole is therefore LGPLv3"):

| Files | License |
| --- | --- |
| `sep.h`, `sepcore.h`, `extract.h`, `analyse.c`, `aperture.c`, `background.c`, `convolve.c`, `deblend.c`, `extract.c`, `lutz.c`; `aperture.i` (no header, included into `aperture.c`) | LGPL-3.0-or-later (Copyright 1993-2011 Emmanuel Bertin, IAP/CNRS/UPMC; 2014 SEP developers) |
| `util.c` | MIT, except `fqcmp()`/`fqmedian()`, which are LGPL-3.0-or-later |
| `overlap.h` | BSD-3-Clause (photutils developers 2011-14) |
| `sep.pyx` (Python wrapper) | MIT (not vendored) |

`licenses/` holds the upstream texts: `LGPL_LICENSE.txt` (LGPLv3; upstream
does not ship the GPLv3 text it incorporates by reference),
`BSD_LICENSE.txt`, `MIT_LICENSE.txt`.

## Build identity

`build.rs` compiles these files with `-DSEP_VERSION_STRING="1.4.1"` (the
in-source fallback in `util.c` is "1.3.6"). The PyPI 1.4.1 macOS arm64 wheel
embeds the string `1.4.2-dev0`, consistent with upstream `setup.py`'s
`setuptools_scm` scheme (`"{guessed}-dev{distance}"`) formatting the next
version at distance 0 from the tag.
