import os
from pathlib import Path
from PySide6 import QtCore, QtGui, QtSvg

def convert_svg_to_pdf(svg_path: Path, pdf_path: Path):
    renderer = QtSvg.QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise ValueError(f"Invalid SVG: {svg_path}")
    sz = renderer.defaultSize()
    
    writer = QtGui.QPdfWriter(str(pdf_path))
    writer.setPageSize(QtGui.QPageSize(QtCore.QSizeF(sz.width(), sz.height()), QtGui.QPageSize.Unit.Point))
    writer.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    writer.setResolution(72)
    
    painter = QtGui.QPainter(writer)
    renderer.render(painter)
    painter.end()
    print(f"Exported: {pdf_path} (size: {pdf_path.stat().st_size} bytes)")

def main():
    app = QtGui.QGuiApplication.instance()
    if not app:
        app = QtGui.QGuiApplication([])
        
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    svg_files = sorted(figures_dir.glob("*.svg"))
    
    print(f"Found {len(svg_files)} SVG figures in {figures_dir}")
    for svg_file in svg_files:
        pdf_file = svg_file.with_suffix(".pdf")
        convert_svg_to_pdf(svg_file, pdf_file)

if __name__ == "__main__":
    main()
