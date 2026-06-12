#!/usr/bin/env python3

from pathlib import Path
import math
import os
import subprocess
import sys


def load_vector(path: Path) -> list[float]:
    values: list[float] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values.extend(float(token) for token in line.split())
    return values


def load_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append([float(token) for token in line.split()])
    return rows


def load_times(path: Path) -> list[float]:
    times: list[float] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[1]))
    return times


def sample_indices(size: int, count: int) -> list[int]:
    if size <= 0:
        return []
    if count >= size:
        return list(range(size))
    indices: list[int] = []
    for i in range(count):
        idx = round(i * (size - 1) / max(count - 1, 1))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


def draw_line(buffer: list[int], width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            buffer[y0 * width + x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def lzw_compress(indices: list[int], min_code_size: int) -> bytes:
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    bit_buffer = 0
    bit_count = 0
    payload = bytearray()

    def write_code(code: int, code_size: int) -> None:
        nonlocal bit_buffer, bit_count
        bit_buffer |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            payload.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8

    def reset_table() -> tuple[dict[tuple[int, ...], int], int, int]:
        table = {(i,): i for i in range(clear_code)}
        return table, end_code + 1, min_code_size + 1

    table, next_code, code_size = reset_table()
    write_code(clear_code, code_size)

    phrase = (indices[0],)
    for value in indices[1:]:
        candidate = phrase + (value,)
        if candidate in table:
            phrase = candidate
            continue

        write_code(table[phrase], code_size)

        if next_code < 4096:
            table[candidate] = next_code
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            write_code(clear_code, code_size)
            table, next_code, code_size = reset_table()

        phrase = (value,)

    write_code(table[phrase], code_size)
    write_code(end_code, code_size)

    if bit_count > 0:
        payload.append(bit_buffer & 0xFF)

    return bytes(payload)


def gif_sub_blocks(payload: bytes) -> bytes:
    blocks = bytearray()
    for start in range(0, len(payload), 255):
        chunk = payload[start:start + 255]
        blocks.append(len(chunk))
        blocks.extend(chunk)
    blocks.append(0)
    return bytes(blocks)


def write_gif(path: Path, width: int, height: int, palette: list[tuple[int, int, int]], frames: list[list[int]], delay_cs: int) -> None:
    color_table_size = 1
    while color_table_size < len(palette):
        color_table_size *= 2
    palette_bytes = bytearray()
    for r, g, b in palette:
        palette_bytes.extend((r, g, b))
    while len(palette_bytes) < color_table_size * 3:
        palette_bytes.extend((0, 0, 0))

    size_bits = max(0, int(math.log2(color_table_size)) - 1)
    packed_field = 0x80 | 0x70 | size_bits  # global color table, 8-bit color resolution
    min_code_size = max(2, int(math.ceil(math.log2(color_table_size))))

    with path.open("wb") as handle:
        handle.write(b"GIF89a")
        handle.write(width.to_bytes(2, "little"))
        handle.write(height.to_bytes(2, "little"))
        handle.write(bytes([packed_field, 0, 0]))
        handle.write(palette_bytes)

        handle.write(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")

        for frame in frames:
            handle.write(b"\x21\xF9\x04\x04")
            handle.write(delay_cs.to_bytes(2, "little"))
            handle.write(b"\x00\x00")
            handle.write(b"\x2C")
            handle.write((0).to_bytes(2, "little"))
            handle.write((0).to_bytes(2, "little"))
            handle.write(width.to_bytes(2, "little"))
            handle.write(height.to_bytes(2, "little"))
            handle.write(b"\x00")
            handle.write(bytes([min_code_size]))
            handle.write(gif_sub_blocks(lzw_compress(frame, min_code_size)))

        handle.write(b"\x3B")


def render_fallback(output_dir: Path) -> int:
    snapshot_dir = output_dir / "snapshots"
    x = load_vector(output_dir / "x_cells.txt")
    v = load_vector(output_dir / "velocity_axis.txt")
    times = load_times(snapshot_dir / "snapshot_times.txt")
    frame_paths = sorted(snapshot_dir.glob("distribution_f_*.txt"))
    frames = [load_matrix(path) for path in frame_paths]

    width = 800
    height = 500
    left = 70
    right = 20
    top = 20
    bottom = 40
    plot_width = width - left - right
    plot_height = height - top - bottom

    sample_ids = sample_indices(len(x), min(6, len(x)))
    palette = [
        (255, 255, 255),  # 0 background
        (0, 0, 0),        # 1 axes
        (210, 210, 210),  # 2 grid
        (13, 8, 135),     # 3
        (84, 3, 160),     # 4
        (139, 10, 165),   # 5
        (193, 45, 99),    # 6
        (240, 96, 39),    # 7
        (249, 201, 50),   # 8
    ]
    curve_colors = [3, 4, 5, 6, 7, 8]
    v_min = v[0]
    v_max = v[-1]

    raster_frames: list[list[int]] = []
    for frame_index, frame in enumerate(frames):
        buffer = [0] * (width * height)

        x_axis_y = height - bottom
        y_axis_x = left
        draw_line(buffer, width, height, y_axis_x, top, y_axis_x, x_axis_y, 1)
        draw_line(buffer, width, height, y_axis_x, x_axis_y, width - right, x_axis_y, 1)

        frame_peak = max(max(row) for row in frame)
        y_max = max(frame_peak * 1.05, 1e-12)

        for color_id, sample_id in zip(curve_colors, sample_ids):
            profile = frame[sample_id]
            previous = None
            for j, c_value in enumerate(v):
                x_pixel = y_axis_x + int(round((c_value - v_min) / max(v_max - v_min, 1e-12) * plot_width))
                y_pixel = x_axis_y - int(round(profile[j] / y_max * plot_height))
                if previous is not None:
                    draw_line(buffer, width, height, previous[0], previous[1], x_pixel, y_pixel, color_id)
                previous = (x_pixel, y_pixel)

        raster_frames.append(buffer)

    out = output_dir / "distribution_f.gif"
    write_gif(out, width, height, palette, raster_frames, delay_cs=25)
    print(out)
    return 0


def render_with_swift(output_dir: Path) -> int:
    script_path = Path(__file__).with_name("generate_distribution_gif.swift")
    module_cache = Path("/private/tmp/codex-swift-module-cache")
    module_cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    completed = subprocess.run(
        ["swift", "-module-cache-path", str(module_cache), str(script_path), str(output_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.strip(), file=sys.stderr)
        if completed.stdout:
            print(completed.stdout.strip(), file=sys.stderr)
        return completed.returncode or 1
    if completed.stdout:
        print(completed.stdout.strip())
    return 0


def render_with_matplotlib(output_dir: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import animation
    import matplotlib.pyplot as plt
    import numpy as np

    snapshot_dir = output_dir / "snapshots"
    x_path = output_dir / "x_cells.txt"
    v_path = output_dir / "velocity_axis.txt"
    times_path = snapshot_dir / "snapshot_times.txt"
    frame_paths = sorted(snapshot_dir.glob("distribution_f_*.txt"))

    x = np.loadtxt(x_path)
    v = np.loadtxt(v_path)
    times = np.loadtxt(times_path)
    if times.ndim == 1:
        times = times[np.newaxis, :]

    frames = [np.loadtxt(path, ndmin=2) for path in frame_paths]
    ids = np.array(sample_indices(len(x), min(6, len(x))), dtype=int)
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(ids)))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    def draw_frame(frame_idx: int) -> None:
        ax.clear()
        frame = frames[frame_idx]
        for color, sample_index in zip(colors, ids):
            ax.plot(v, frame[sample_index, :], color=color, linewidth=2.0, label=f"x={x[sample_index]:.2f}")

        time_value = times[min(frame_idx, len(times) - 1), 1]
        ax.set_title(f"Distribution f(c) at Selected x, t={time_value:.4f}")
        ax.set_xlabel("velocity c")
        ax.set_ylabel("distribution f")
        ax.set_xlim(float(v[0]), float(v[-1]))
        frame_peak = max(float(np.max(frame)), 1e-12)
        ax.set_ylim(0.0, 1.05 * frame_peak)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=260, repeat=True)
    out = output_dir / "distribution_f.gif"
    anim.save(out, writer=animation.PillowWriter(fps=4))
    print(out)
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 animate_distribution_x_v.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    snapshot_dir = output_dir / "snapshots"
    required = [
        snapshot_dir,
        output_dir / "x_cells.txt",
        output_dir / "velocity_axis.txt",
        snapshot_dir / "snapshot_times.txt",
    ]
    if any(not path.exists() for path in required):
        print("Missing required files for animation.", file=sys.stderr)
        return 1

    if not list(snapshot_dir.glob("distribution_f_*.txt")):
        print("No snapshot frames found.", file=sys.stderr)
        return 1

    try:
        return render_with_matplotlib(output_dir)
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"matplotlib animation failed, falling back to native GIF writer: {exc}", file=sys.stderr)

    swift_status = render_with_swift(output_dir)
    if swift_status == 0:
        return 0

    print("native GIF writer failed, falling back to built-in GIF writer", file=sys.stderr)
    return render_fallback(output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
