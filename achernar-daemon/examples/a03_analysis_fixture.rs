//! JSON-lines bridge for independently generated A03 numerical fixtures.
//! This example invokes the production analysis module without camera hardware.

#[path = "../src/processing/debayer.rs"]
mod debayer;

use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Default, Deserialize)]
struct InputMasks {
    bad_pixel: Option<Vec<u8>>,
    invalid: Option<Vec<u8>>,
    native_saturation: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct Input {
    id: String,
    descriptor: debayer::SensorDescriptor,
    data: Vec<u8>,
    #[serde(default)]
    masks: InputMasks,
    #[serde(default = "include_rgb_default")]
    include_rgb: bool,
}

fn include_rgb_default() -> bool {
    true
}

fn evaluate(value: Value) -> Value {
    let id = value.get("id").cloned().unwrap_or(Value::Null);
    let result = (|| -> anyhow::Result<Value> {
        let input: Input = serde_json::from_value(value)?;
        let frame = debayer::SensorFrame {
            descriptor: &input.descriptor,
            data: &input.data,
            masks: debayer::NativeMasks {
                bad_pixel: input.masks.bad_pixel.as_deref(),
                invalid: input.masks.invalid.as_deref(),
                native_saturation: input.masks.native_saturation.as_deref(),
            },
        };
        let analysis = debayer::analyze(&frame)?;
        let rgb = if input.include_rgb {
            Some(debayer::linear_rgb(&frame)?)
        } else {
            None
        };
        Ok(json!({
            "id": input.id,
            "ok": true,
            "pixels": analysis.pixels,
            "rgb": rgb,
            "masks": analysis.masks,
            "descriptor": analysis.descriptor,
            "geometry_checks": {
                "mapped_origin": input.descriptor.geometry.map_to_sensor(0.0, 0.0),
                "mapped_unit_x": input.descriptor.geometry.map_to_sensor(1.0, 0.0),
                "mapped_unit_y": input.descriptor.geometry.map_to_sensor(0.0, 1.0),
                "isotropic_native_scale": input.descriptor.geometry.isotropic_native_scale(),
            },
        }))
    })();
    match result {
        Ok(output) => output,
        Err(error) => json!({"id": id, "ok": false, "error": format!("{error:#}")}),
    }
}

fn main() -> anyhow::Result<()> {
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(value) => evaluate(value),
            Err(error) => json!({"id": null, "ok": false, "error": error.to_string()}),
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}
