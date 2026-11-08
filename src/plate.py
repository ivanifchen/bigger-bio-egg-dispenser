"""Print jobs as Bambu Studio project 3MFs, written with the Bambu Studio command line:
X2D 0.4 nozzle, stock 0.20mm Standard, Bambu PLA Matte on the main (left) hotend, Textured PEI.
One 3MF per print job. Unsliced: open it in Bambu Studio, pick the AMS slot, slice, print.

Changes from the stock process are per job (JOBS). The Dispenser's halves follow the print
settings agreed for them: 3 walls, 10% gyroid, 8 mm brim, 1.0 mm Z-hop.
"""

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

BAMBU = Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio")
PROFILES = Path("/Applications/BambuStudio.app/Contents/Resources/profiles/BBL")
PRESETS = {
    "machine": "Bambu Lab X2D 0.4 nozzle",
    "process": "0.20mm Standard @BBL X2D",
    "filament": "Bambu PLA Matte @BBL X2D 0.4 nozzle",
}
BED = 256.0
MAIN_HOTEND = {"filament_map_mode": "Manual", "filament_map": ["1"]}

JOBS = {
    "coupon": {
        "machine": {"curr_bed_type": "Textured PEI Plate"},
        "process": {"wall_loops": "3", "sparse_infill_density": "25%",
                    "sparse_infill_pattern": "gyroid", **MAIN_HOTEND},
    },
    "half": {
        "machine": {"curr_bed_type": "Textured PEI Plate", "z_hop": ["1"]},
        "process": {"wall_loops": "3", "sparse_infill_density": "10%",
                    "sparse_infill_pattern": "gyroid", "brim_type": "outer_only",
                    "brim_width": "8", "enable_support": "0", **MAIN_HOTEND},
    },
}


def _flat_preset(kind, name):
    """A system preset with its `inherits` chain merged in; the CLI does not resolve it."""
    data = json.loads((PROFILES / kind / f"{name}.json").read_text())
    if data.get("inherits"):
        merged = _flat_preset(kind, data["inherits"])
        merged.update(data)
        data = merged
    data.pop("inherits", None)
    data["name"] = name
    return data


def check_on_bed(parts):
    for label, mesh in parts:
        lo, hi = mesh.bounds
        assert lo[0] >= 0 and lo[1] >= 0 and hi[0] <= BED and hi[1] <= BED, (label, lo, hi)
        assert abs(lo[2]) < 1e-3, (label, lo[2])
        assert hi[2] <= 256.0, (label, "too tall for the X2D", hi[2])


# Above this print height, the halves' walls slow down: the designer's warning for fast printers
# (the Original's own profile halves them above 150 mm on a P1S; the X2D is faster still).
SLOW_ABOVE = 70.0
SLOW = {"outer_wall_speed": "100", "inner_wall_speed": "150"}


def add_height_ranges(path, ranges):
    """Give objects of a written project 3MF per-height settings, as Bambu Studio's
    Metadata/layer_config_ranges.xml (the CLI can't). ranges: {label: [(z0, z1, {opt: value})]}.
    Objects are numbered by their order in model_settings.config, as Bambu Studio does."""
    import re
    import shutil
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = re.findall(r'<object id="\d+">\s*<metadata key="name" value="([^"]+)"',
                           z.read("Metadata/model_settings.config").decode())
        items = {n: z.read(n) for n in z.namelist()}
    xml = ['<?xml version="1.0" encoding="utf-8"?>', "<objects>"]
    for index, name in enumerate(names, start=1):
        label = name.rsplit(".", 1)[0]
        if label not in ranges:
            continue
        xml.append(f' <object id="{index}">')
        for z0, z1, opts in ranges[label]:
            xml.append(f'  <range min_z="{z0:g}" max_z="{z1:g}">')
            for k, v in {"extruder": "0", "layer_height": "0.2", **opts}.items():
                xml.append(f'   <option opt_key="{k}">{v}</option>')
            xml.append("  </range>")
        xml.append(" </object>")
    xml.append("</objects>")
    items["Metadata/layer_config_ranges.xml"] = "\n".join(xml).encode()
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in items.items():
            z.writestr(n, data)
    shutil.move(tmp, path)


def slice_project(path):
    """Slice a project 3MF as it stands (its own settings and height ranges) in a temporary
    folder; return (returncode, slice_info)."""
    path = Path(path).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sliced.gcode.3mf"
        run = subprocess.run([str(BAMBU), "--slice", "0", "--outputdir", tmp, "--export-3mf",
                              out.name, str(path)], capture_output=True, text=True)
        info = ""
        if out.exists():
            with zipfile.ZipFile(out) as z:
                info = z.read("Metadata/slice_info.config").decode()
        return run.returncode, info


def write_project(path, parts, job, slice_check=False):
    """Write `parts` [(label, trimesh in bed coordinates)] as a Bambu Studio project 3MF.
    With slice_check, also slice it in a temporary folder and return (returncode, slice_info)."""
    check_on_bed(parts)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stls = []
        for label, mesh in parts:
            stl = tmp / f"{label}.stl"
            mesh.export(stl)
            stls.append(str(stl))
        files = {}
        for kind, name in PRESETS.items():
            data = _flat_preset(kind, name)
            data.update(JOBS[job].get(kind, {}))
            files[kind] = tmp / f"{kind}.json"
            files[kind].write_text(json.dumps(data))
        base = [str(BAMBU), "--load-settings", f"{files['machine']};{files['process']}",
                "--load-filaments", str(files["filament"]), "--arrange", "0", "--orient", "0"]
        # --outputdir also prefixes --export-3mf, so pass the bare name.
        run = subprocess.run(base + ["--outputdir", str(path.parent), "--export-3mf", path.name,
                                     *stls], capture_output=True, text=True)
        (path.parent / "result.json").unlink(missing_ok=True)
        assert path.exists(), f"Bambu Studio wrote no 3MF:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}"
        if slice_check:
            sliced = tmp / "sliced.gcode.3mf"
            run = subprocess.run(base + ["--slice", "0", "--outputdir", str(tmp),
                                         "--export-3mf", sliced.name, *stls],
                                 capture_output=True, text=True)
            info = ""
            if sliced.exists():
                with zipfile.ZipFile(sliced) as z:
                    info = z.read("Metadata/slice_info.config").decode()
            return run.returncode, info
    return 0, ""


def slice_summary(info):
    """Print time (h) and filament (g) from a slice_info.config."""
    import re
    t = re.search(r'key="prediction" value="(\d+)"', info)
    g = re.search(r'key="weight" value="([\d.]+)"', info)
    return (int(t.group(1)) / 3600 if t else None, float(g.group(1)) if g else None)
