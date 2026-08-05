#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Achernar DSLR / mirrorless bench check.
#
# libgphoto2's device list says a body is *recognised*; it does not say which
# controls that body exposes, what its shutter and ISO ladders look like, or
# whether it will give you live view. The gPhoto project is explicit about
# this — the controls of each individual camera have to be enumerated on the
# camera. This script does that enumeration and writes a report, so support for
# a body is a measured fact rather than an assumption.
#
# Run it once per camera model, with the body connected by USB, powered on,
# and — this matters — with any vendor tethering software quit, since macOS
# PTPCamera and Canon EOS Utility both claim the device exclusively.
#
# Usage:   ./scripts/dslr_bench.sh [output_dir]
# Output:  <output_dir>/<model>-bench.txt  and  <model>-config.txt
#
# Exit status is 0 when the body is usable by the daemon, 1 when a hard
# blocker was found (no camera, no RAW decoder, no exposure control).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

OUT_DIR="${1:-./dslr-bench}"
mkdir -p "$OUT_DIR"

BLOCKERS=0
WARNINGS=0

note()  { printf '  %s\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
bad()   { printf '  \033[31m✗\033[0m %s\n' "$*"; BLOCKERS=$((BLOCKERS + 1)); }
head1() { printf '\n\033[1m%s\033[0m\n' "$*"; }

printf '╔═══════════════════════════════════════════════╗\n'
printf '║   Achernar — DSLR / mirrorless bench check    ║\n'
printf '╚═══════════════════════════════════════════════╝\n'

# ─── 1. Host libraries ───────────────────────────────────────────────────────
head1 "1. Host libraries"

if command -v gphoto2 >/dev/null 2>&1; then
    ok "gphoto2 CLI $(gphoto2 --version 2>/dev/null | head -1 | awk '{print $NF}')"
else
    bad "gphoto2 CLI not found (macOS: brew install gphoto2)"
fi

if pkg-config --exists libgphoto2 2>/dev/null; then
    ok "libgphoto2 $(pkg-config --modversion libgphoto2)"
else
    bad "libgphoto2 development files not found — the daemon cannot build the DSLR driver"
fi

# LibRaw is the difference between a camera that streams and one that connects
# and then produces nothing. Canon RAW-only (CR2/CR3) and Panasonic RW2 have no
# other decode path.
if pkg-config --exists libraw 2>/dev/null; then
    ok "LibRaw $(pkg-config --modversion libraw)"
elif ls /opt/homebrew/lib/libraw* /usr/local/lib/libraw* /usr/lib/libraw* >/dev/null 2>&1; then
    ok "LibRaw present (no pkg-config file; CMake will use the library fallback)"
else
    bad "LibRaw not found — RAW capture will fail (macOS: brew install libraw)"
fi

# ─── 2. Conflicting processes ────────────────────────────────────────────────
head1 "2. Exclusive-access conflicts"

CONFLICTS=0
for proc in PTPCamera "EOS Utility" "Image Capture" gvfs-gphoto2-volume-monitor gvfsd-gphoto2; do
    if pgrep -f "$proc" >/dev/null 2>&1; then
        warn "\"$proc\" is running and will hold the camera open — quit it first"
        CONFLICTS=$((CONFLICTS + 1))
    fi
done
[ "$CONFLICTS" -eq 0 ] && ok "no known tethering process holding the USB device"

# ─── 3. Detection ────────────────────────────────────────────────────────────
head1 "3. Camera detection"

if ! command -v gphoto2 >/dev/null 2>&1; then
    printf '\nCannot continue without the gphoto2 CLI.\n'
    exit 1
fi

DETECT="$(gphoto2 --auto-detect 2>&1)"
printf '%s\n' "$DETECT" | sed 's/^/    /'

# Skip the two header lines gphoto2 prints before the table.
MODEL="$(printf '%s\n' "$DETECT" | tail -n +3 | head -1 | sed 's/  *usb:.*$//' | sed 's/[[:space:]]*$//')"

if [ -z "$MODEL" ]; then
    bad "no camera detected — check USB cable, power, and that the body is not in playback-only mode"
    printf '\n%d blocker(s), %d warning(s)\n' "$BLOCKERS" "$WARNINGS"
    exit 1
fi
ok "detected: $MODEL"

SAFE_MODEL="$(printf '%s' "$MODEL" | tr ' /' '__')"
REPORT="$OUT_DIR/${SAFE_MODEL}-bench.txt"
CONFIG="$OUT_DIR/${SAFE_MODEL}-config.txt"

{
    printf 'Achernar DSLR bench report\n'
    printf 'model: %s\n' "$MODEL"
    printf 'date:  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host:  %s %s\n\n' "$(uname -s)" "$(uname -m)"
} > "$REPORT"

gphoto2 --summary >> "$REPORT" 2>&1

# ─── 4. Control surface ──────────────────────────────────────────────────────
head1 "4. Control surface"

gphoto2 --list-config > "$CONFIG" 2>&1
note "full config tree written to $CONFIG ($(wc -l < "$CONFIG" | tr -d ' ') entries)"

# These are exactly the keys src/gphoto2_camera.cpp probes in
# supported_options(). If a body names them differently, the driver's
# opt_to_name() mapping needs a per-model case — which is the whole reason for
# enumerating rather than assuming.
probe_config() {
    local key="$1" required="$2" label="$3"
    if grep -qx "/main/[a-z]*/[a-z]*/${key}" "$CONFIG" 2>/dev/null || grep -q "/${key}\$" "$CONFIG" 2>/dev/null; then
        local dump
        dump="$(gphoto2 --get-config "$key" 2>&1)"
        local current
        current="$(printf '%s\n' "$dump" | awk -F': ' '/^Current: /{print $2; exit}')"
        local nchoices
        nchoices="$(printf '%s\n' "$dump" | grep -c '^Choice:')"
        ok "$label: present, current=\"${current:-?}\", ${nchoices} choices"
        {
            printf '\n===== %s =====\n' "$key"
            printf '%s\n' "$dump"
        } >> "$REPORT"
        return 0
    fi
    if [ "$required" = "required" ]; then
        bad "$label: NOT exposed by this body (config key \"$key\" missing)"
    else
        warn "$label: not exposed (config key \"$key\" missing)"
    fi
    return 1
}

probe_config shutterspeed  required "shutter ladder"
probe_config iso           required "ISO ladder"
probe_config viewfinder    optional "viewfinder toggle"
probe_config capturetarget optional "capture target"
probe_config imageformat   optional "image format"

# ─── 5. Shutter/ISO label sanity ─────────────────────────────────────────────
# The daemon parses these labels into physical values (hardware/dslr_controls.rs).
# Any label it cannot parse becomes an exposure it will refuse to select, so
# surface them here rather than discovering it at the telescope.
head1 "5. Label parseability"

check_labels() {
    local key="$1" kind="$2"
    local labels
    labels="$(gphoto2 --get-config "$key" 2>/dev/null | sed -n 's/^Choice: [0-9]* //p')"
    [ -z "$labels" ] && { warn "$key: no choices returned"; return; }

    local total=0 parsed=0 unparsed=""
    while IFS= read -r l; do
        [ -z "$l" ] && continue
        total=$((total + 1))
        if [ "$kind" = shutter ]; then
            # fraction, decimal, or integer, with an optional " or s suffix
            printf '%s' "$l" | grep -Eq '^[0-9]+(\.[0-9]+)?(/[0-9]+)?["s]?$' \
                && parsed=$((parsed + 1)) || unparsed="$unparsed \"$l\""
        else
            printf '%s' "$l" | grep -Eq '^(ISO ?)?[0-9]+$' \
                && parsed=$((parsed + 1)) || unparsed="$unparsed \"$l\""
        fi
    done <<EOF
$labels
EOF

    if [ "$parsed" -eq 0 ]; then
        bad "$key: none of $total labels are numeric — the daemon cannot set an exposure"
    else
        ok "$key: $parsed/$total labels parse to physical values"
        [ -n "$unparsed" ] && note "non-numeric (expected for bulb/auto):$unparsed"
    fi
    printf '\n===== %s parseability: %s/%s =====\n%s\n' \
        "$key" "$parsed" "$total" "$labels" >> "$REPORT"
}

check_labels shutterspeed shutter
check_labels iso iso

# ─── 6. Live view ────────────────────────────────────────────────────────────
# Index 0 of the driver's format list. If this fails, every stream on this body
# costs a shutter actuation per frame.
head1 "6. Live view (zero shutter actuations)"

PREVIEW="$OUT_DIR/${SAFE_MODEL}-preview.jpg"
rm -f "$PREVIEW"
if gphoto2 --capture-preview --force-overwrite --filename "$PREVIEW" >/dev/null 2>&1 \
   && [ -s "$PREVIEW" ]; then
    SIZE="$(wc -c < "$PREVIEW" | tr -d ' ')"
    DIMS="$(command -v identify >/dev/null 2>&1 && identify -format '%wx%h' "$PREVIEW" 2>/dev/null || echo unknown)"
    ok "live view works — ${SIZE} bytes, ${DIMS} ($PREVIEW)"
    printf '\n===== liveview: OK %s bytes %s =====\n' "$SIZE" "$DIMS" >> "$REPORT"
else
    warn "live view unavailable — framing will fire the shutter once per frame"
    printf '\n===== liveview: UNAVAILABLE =====\n' >> "$REPORT"
fi

# ─── 7. Set-and-read-back ────────────────────────────────────────────────────
# Enumerating a ladder proves the body *lists* those values. It does not prove
# the body will accept one. Bodies in certain exposure modes silently refuse
# config writes, and the daemon reads settings back for exactly this reason —
# so the bench has to exercise the same round trip.
head1 "7. Settings round trip"

roundtrip() {
    local key="$1"
    local before after target
    before="$(gphoto2 --get-config "$key" 2>/dev/null | awk -F': ' '/^Current: /{print $2; exit}')"
    [ -z "$before" ] && { warn "$key: no current value; skipping round trip"; return; }

    # Pick a different choice than the current one, then put it back.
    target="$(gphoto2 --get-config "$key" 2>/dev/null | sed -n 's/^Choice: [0-9]* //p' \
              | grep -v -x -F "$before" | grep -Ev '^(bulb|Bulb|BULB|Auto|auto)$' | head -1)"
    [ -z "$target" ] && { warn "$key: only one settable choice; skipping round trip"; return; }

    if ! gphoto2 --set-config-value "$key=$target" >/dev/null 2>&1; then
        bad "$key: camera refused a write (tried \"$target\") — check the mode dial is on M"
        return
    fi
    after="$(gphoto2 --get-config "$key" 2>/dev/null | awk -F': ' '/^Current: /{print $2; exit}')"
    if [ "$after" = "$target" ]; then
        ok "$key: set \"$target\" and read it back"
    else
        bad "$key: wrote \"$target\" but the camera reports \"$after\" — writes are being ignored"
    fi
    gphoto2 --set-config-value "$key=$before" >/dev/null 2>&1 || true
    printf '\n===== %s round trip: wrote %s, read %s =====\n' "$key" "$target" "$after" >> "$REPORT"
}

roundtrip shutterspeed
roundtrip iso

# ─── 8. RAW capture round trip ───────────────────────────────────────────────
# The point of this section is to prove the daemon's decode path works on THIS
# body. A previous version accepted a JPEG and reported "Usable.", so a camera
# configured for JPEG passed a test named "RAW capture round trip" without a
# single RAW file ever existing. It now selects RAW explicitly and fails if no
# RAW arrives or none decodes.
head1 "8. RAW capture round trip"

RAW_MODE_SET=0
if grep -q "imageformat" "$CONFIG" 2>/dev/null; then
    ORIGINAL_FORMAT="$(gphoto2 --get-config imageformat 2>/dev/null \
                       | awk -F': ' '/^Current: /{print $2; exit}')"
    # Prefer RAW+JPEG where offered — that is the shipping default on the R50
    # and M50, and it is the configuration most likely to expose companion-file
    # bugs. Fall back to plain RAW.
    RAW_CHOICE="$(gphoto2 --get-config imageformat 2>/dev/null | sed -n 's/^Choice: [0-9]* //p' \
                  | grep -iE 'raw' | grep -iE 'jpe?g|\+' | head -1)"
    [ -z "$RAW_CHOICE" ] && RAW_CHOICE="$(gphoto2 --get-config imageformat 2>/dev/null \
                  | sed -n 's/^Choice: [0-9]* //p' | grep -iE '^\s*raw\s*$|raw' | head -1)"

    if [ -n "$RAW_CHOICE" ]; then
        if gphoto2 --set-config-value "imageformat=$RAW_CHOICE" >/dev/null 2>&1; then
            ok "image format set to \"$RAW_CHOICE\" for this test"
            RAW_MODE_SET=1
        else
            bad "could not select a RAW image format (\"$RAW_CHOICE\" refused)"
        fi
    else
        bad "this body lists no RAW image format — the daemon cannot capture RAW from it"
    fi
else
    warn "imageformat not exposed; capturing in whatever mode the body is set to"
fi

printf '  This fires the shutter once. Ctrl-C within 5s to skip.\n'
sleep 5

RAW_BASE="$OUT_DIR/${SAFE_MODEL}-capture"
rm -f "${RAW_BASE}".*
RAW_SEEN=0
RAW_DECODED=0

if gphoto2 --capture-image-and-download --force-overwrite \
        --filename "${RAW_BASE}.%C" >/dev/null 2>&1; then
    FOUND="$(ls "${RAW_BASE}".* 2>/dev/null || true)"
    if [ -z "$FOUND" ]; then
        bad "capture reported success but downloaded no file"
    fi
    for f in $FOUND; do
        EXT="$(printf '%s' "${f##*.}" | tr 'A-Z' 'a-z')"
        SZ="$(wc -c < "$f" | tr -d ' ')"
        ok "downloaded .$EXT ($SZ bytes)"
        printf '\n===== capture: %s %s bytes =====\n' "$EXT" "$SZ" >> "$REPORT"
        case "$EXT" in
            jpg|jpeg)
                note "JPEG companion present (expected in RAW+JPEG)"
                ;;
            *)
                RAW_SEEN=1
                # Confirm the decoder the daemon will use can actually read
                # this body's RAW, not merely that a RAW arrived.
                if command -v raw-identify >/dev/null 2>&1; then
                    if raw-identify "$f" >/dev/null 2>&1; then
                        ok "LibRaw decodes .$EXT from this body"
                        RAW_DECODED=1
                    else
                        bad "LibRaw cannot decode .$EXT from this body — update LibRaw"
                    fi
                elif command -v dcraw_emu >/dev/null 2>&1; then
                    if dcraw_emu -i "$f" >/dev/null 2>&1; then
                        ok "LibRaw (dcraw_emu) decodes .$EXT from this body"
                        RAW_DECODED=1
                    else
                        bad "LibRaw cannot decode .$EXT from this body"
                    fi
                else
                    bad "no LibRaw CLI (raw-identify / dcraw_emu) available to verify .$EXT decodes \
— install LibRaw's tools; an unverified decoder is not a tested one"
                fi
                ;;
        esac
    done
else
    bad "capture failed — check the mode dial is not in a fully automatic scene mode"
fi

# Restore whatever the operator had set.
if [ "$RAW_MODE_SET" -eq 1 ] && [ -n "${ORIGINAL_FORMAT:-}" ]; then
    gphoto2 --set-config-value "imageformat=$ORIGINAL_FORMAT" >/dev/null 2>&1 \
        && note "image format restored to \"$ORIGINAL_FORMAT\"" \
        || warn "could not restore image format to \"$ORIGINAL_FORMAT\""
fi

if [ "$RAW_SEEN" -eq 0 ]; then
    bad "no RAW file was produced — this body has not been shown to work for science frames"
elif [ "$RAW_DECODED" -eq 0 ]; then
    bad "a RAW arrived but nothing could decode it"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
head1 "Summary"
note "model:  $MODEL"
note "report: $REPORT"
note "config: $CONFIG"

if [ "$BLOCKERS" -gt 0 ]; then
    printf '\n\033[31m%d blocker(s)\033[0m, %d warning(s) — this body is not yet usable.\n' \
        "$BLOCKERS" "$WARNINGS"
    exit 1
fi
printf '\n\033[32mUsable.\033[0m %d warning(s).\n' "$WARNINGS"
printf 'Attach %s to the PR that claims support for this model.\n' "$(basename "$REPORT")"
exit 0
