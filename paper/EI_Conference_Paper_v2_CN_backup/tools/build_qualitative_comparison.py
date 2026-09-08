#!/usr/bin/env python3
"""Deterministically compose the full-protocol qualitative comparison.

No inference, metric computation, prediction modification, cropping, or generative processing.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "paper/OmniSAM_Paper_v1_0_CN"
AUDIT = PAPER / "notes/baseline_qualitative_export_audit.md"
OUTPUT = PAPER / "figures/fig_qualitative_full_protocol.png"
SAMPLES = (24, 68, 101, 111)
COLUMNS = ("Input", "Ground Truth", "OmniSAM Baseline", "Ours (MTC+SMR)")
CELL_W, CELL_H = 2048, 400
LEFT_MARGIN, COL_GAP, ROW_GAP = 250, 20, 24
HEADER_H = 90
BG = (255, 255, 255)
TEXT = (20, 20, 20)


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def mapped_paths(sid):
    # Mapping is fixed by baseline_qualitative_export_audit.md; require its exact evidence lines.
    input_rel = f"data/DensePASS/leftImg8bit/val/{sid}_.png"
    gt_rel = f"data/DensePASS/gtFine/val/{sid}_labelTrainIds.png"
    baseline_rel = f"paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_{sid:03d}/pred_color.png"
    ours_rel = f"paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_{sid:03d}/pred_color.png"
    audit_text = AUDIT.read_text(encoding="utf-8")
    for rel in (input_rel, gt_rel, baseline_rel, ours_rel):
        if rel not in audit_text:
            raise RuntimeError(f"path is not documented by audit: {rel}")
    return tuple(REPO / rel for rel in (input_rel, gt_rel, baseline_rel, ours_rel))


def load_exact(path):
    image = Image.open(path).convert("RGB")
    if image.size != (CELL_W, CELL_H):
        raise RuntimeError(f"unexpected size {image.size}: {path}")
    return image


def centered_text(draw, box, text, text_font):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=text_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1-x0-tw)/2, y0 + (y1-y0-th)/2 - bbox[1]), text, fill=TEXT, font=text_font)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    title_font = font(42, bold=True)
    row_font = font(38, bold=True)
    width = LEFT_MARGIN + 4 * CELL_W + 3 * COL_GAP
    height = HEADER_H + 4 * CELL_H + 3 * ROW_GAP
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    for col, title in enumerate(COLUMNS):
        x = LEFT_MARGIN + col * (CELL_W + COL_GAP)
        centered_text(draw, (x, 0, x + CELL_W, HEADER_H), title, title_font)

    for row, sid in enumerate(SAMPLES):
        y = HEADER_H + row * (CELL_H + ROW_GAP)
        centered_text(draw, (0, y, LEFT_MARGIN - 15, y + CELL_H), f"Sample {sid}", row_font)
        for col, path in enumerate(mapped_paths(sid)):
            x = LEFT_MARGIN + col * (CELL_W + COL_GAP)
            canvas.paste(load_exact(path), (x, y))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", compress_level=6, dpi=(300, 300))
    check = Image.open(OUTPUT)
    if check.size != (width, height) or check.mode != "RGB":
        raise RuntimeError("saved output verification failed")
    print(f"OUTPUT={OUTPUT}")
    print(f"SIZE={width}x{height}")
    print(f"BYTES={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
