"""True-size vector A4 targets generated from the detector's board definition."""

from io import BytesIO

import cv2
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from .config import BoardConfig


def pdf(config: BoardConfig) -> bytes:
    width = config.columns * config.square_mm
    height = config.rows * config.square_mm
    if width > 277 or height > 190:
        raise ValueError("This pattern does not fit A4 landscape with 10 mm margins.")
    board = config.create()
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=landscape(A4), invariant=1)
    canvas.setTitle(f"ChArUco {config.columns}x{config.rows} - {config.square_mm:g} mm squares")
    origin_x = (297 - width) / 2
    origin_y = (210 - height) / 2
    layout = board.generateImage((config.columns * 40, config.rows * 40))

    def rectangle(x, y, w, h):
        canvas.rect(
            (origin_x + x) * mm, (origin_y + height - y - h) * mm, w * mm, h * mm, fill=1, stroke=0
        )

    canvas.setFillColorRGB(0, 0, 0)
    for row in range(config.rows):
        for col in range(config.columns):
            if layout[row * 40 + 1, col * 40 + 1] == 0:
                rectangle(
                    col * config.square_mm,
                    row * config.square_mm,
                    config.square_mm,
                    config.square_mm,
                )
    dictionary = board.getDictionary()
    modules = dictionary.markerSize + 2
    cell = config.marker_mm / modules
    for marker_id, obj in zip(board.getIds().reshape(-1), board.getObjPoints()):
        raster = cv2.aruco.generateImageMarker(dictionary, int(marker_id), modules)
        x, y = obj[:, :2].min(axis=0)
        for row in range(modules):
            for col in range(modules):
                if raster[row, col] == 0:
                    rectangle(float(x) + col * cell, float(y) + row * cell, cell, cell)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        148.5 * mm,
        203 * mm,
        f"ChArUco {config.columns} x {config.rows} squares | square {config.square_mm:g} mm | "
        f"marker {config.marker_mm:g} mm | pattern {width:g} x {height:g} mm",
    )
    canvas.setFont("Helvetica", 7)
    canvas.drawString(
        15 * mm, 8 * mm, "A4 landscape. Print at 100% / Actual size. Measure before use."
    )
    canvas.drawString(
        15 * mm,
        4.5 * mm,
        f"{config.dictionary} | {'legacy' if config.legacy else 'modern'} layout | Mount flat on rigid backing.",
    )
    canvas.setLineWidth(0.5)
    canvas.line(181 * mm, 9 * mm, 281 * mm, 9 * mm)
    for x in range(181, 282, 10):
        canvas.line(x * mm, 7.5 * mm, x * mm, 10.5 * mm)
    canvas.drawCentredString(231 * mm, 4.5 * mm, "100 mm reference - check with a ruler")
    canvas.showPage()
    canvas.save()
    return stream.getvalue()
