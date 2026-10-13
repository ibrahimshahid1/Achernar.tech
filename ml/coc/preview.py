"""Deterministic validation preview: input, partial human regions, model map."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ml.coc.checkpoint import load_checkpoint
from ml.coc.data import HumanCloudDataset


def preview(checkpoint, human_root, output, count=4):
    output = Path(output)
    if output.exists() or count < 1:
        raise ValueError("Use a fresh preview output and positive count")
    model, state = load_checkpoint(checkpoint)
    data = HumanCloudDataset(human_root, "validation", size=state["size"])
    indices = np.linspace(0, len(data) - 1, min(count, len(data)), dtype=int)
    width, margin, row_height = 256, 14, 300
    canvas = Image.new("RGB", (3 * width + 2 * margin, 90 + len(indices) * row_height), "#202630")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    draw.text((margin, 8), "COC research preview | evenly spaced validation records", font=font, fill="white")
    draw.text((margin, 32), "Blue: sky  White: cloud  Orange: contamination  Gray: ignored", font=font, fill="white")
    colors = np.array([[50, 120, 220], [240, 240, 240], [230, 110, 40]], dtype=np.uint8)
    for row, index in enumerate(indices):
        item = data[int(index)]
        image, target = item["image"].numpy(), item["mask"].numpy()
        with torch.inference_mode():
            prediction = model(item["image"][None]).argmax(1)[0].numpy()
        rgb = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
        annotated = np.full_like(rgb, 90)
        keep = target != 255
        annotated[keep] = colors[target[keep]]
        panels = [rgb, annotated, colors[prediction]]
        y = 82 + row * row_height
        draw.text((margin, y), f"{item['sample_id']} | annotated {keep.mean():.1%}", font=font, fill="white")
        for column, (pixels, label) in enumerate(zip(panels, ["Model RGB input", "Human partial labels", "Predicted classes"])):
            x = margin + column * width
            draw.text((x, y + 20), label, font=font, fill="white")
            canvas.paste(Image.fromarray(pixels).resize((width, width), Image.Resampling.NEAREST), (x, y + 40))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return [data.records[int(i)]["sample_id"] for i in indices]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--human-root", default="ml/data/public/lenghu_sky8_human")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=4)
    a = p.parse_args()
    print(preview(a.checkpoint, a.human_root, a.output, a.count))


if __name__ == "__main__":
    main()
