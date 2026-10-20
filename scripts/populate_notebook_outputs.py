"""
Populate notebook outputs by executing code cells in an isolated Python environment
and inserting standard text, display outputs, and base64 figures into the notebook.
"""
import os
import sys
import io
import contextlib
import base64
import json
import nbformat

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if os.path.exists('debug_error.txt'):
    os.remove('debug_error.txt')

from pathlib import Path

nb_path = str(Path(__file__).resolve().parent / 'road_level_flood_pred.ipynb')
with open(nb_path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Loaded notebook with {len(nb.cells)} cells.")

captured_displays = []

def custom_display(*args, **kwargs):
    for arg in args:
        captured_displays.append(arg)

global_scope = {
    '__name__': '__main__',
    'display': custom_display
}

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f"Executing Code Cell {idx}...", flush=True)
        code = cell.source
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                exec(code, global_scope)
            except Exception as e:
                import traceback
                traceback.print_exc(file=stdout_buf)
                with open('debug_error.txt', 'a', encoding='utf-8') as ef:
                    ef.write(f"Error in cell {idx}: {e}\\n")
                    traceback.print_exc(file=ef)
                
        out_text = stdout_buf.getvalue()
        err_text = stderr_buf.getvalue()
        
        cell.outputs = []
        if out_text:
            cell.outputs.append(nbformat.v4.new_output(
                output_type='stream',
                name='stdout',
                text=out_text
            ))
        if err_text:
            cell.outputs.append(nbformat.v4.new_output(
                output_type='stream',
                name='stderr',
                text=err_text
            ))
        while captured_displays:
            disp_obj = captured_displays.pop(0)
            data_dict = {}
            if hasattr(disp_obj, 'data'):
                # SVG or similar
                if hasattr(disp_obj, '_repr_svg_'):
                    data_dict['image/svg+xml'] = disp_obj._repr_svg_()
                elif hasattr(disp_obj, '_repr_html_'):
                    data_dict['text/html'] = disp_obj._repr_html_()
                else:
                    data_dict['text/plain'] = str(disp_obj)
            elif hasattr(disp_obj, '_repr_html_'):
                data_dict['text/html'] = disp_obj._repr_html_()
            else:
                data_dict['text/plain'] = str(disp_obj)
            
            cell.outputs.append(nbformat.v4.new_output(
                output_type='display_data',
                data=data_dict
            ))
            
        cell.execution_count = idx

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook successfully executed and populated with all cell outputs!")
