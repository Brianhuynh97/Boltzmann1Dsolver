#!/usr/bin/env python3

from pathlib import Path
import json
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
    out: list[int] = []
    for i in range(count):
        idx = round(i * (size - 1) / max(count - 1, 1))
        if not out or idx != out[-1]:
            out.append(idx)
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 generate_distribution_animation_html.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    snapshot_dir = output_dir / "snapshots"
    x_path = output_dir / "x_cells.txt"
    v_path = output_dir / "velocity_axis.txt"
    times_path = snapshot_dir / "snapshot_times.txt"
    frame_paths = sorted(snapshot_dir.glob("distribution_f_*.txt"))

    required = [snapshot_dir, x_path, v_path, times_path]
    if any(not path.exists() for path in required) or not frame_paths:
        print("Missing required files for HTML animation.", file=sys.stderr)
        return 1

    x = load_vector(x_path)
    v = load_vector(v_path)
    times = load_times(times_path)
    frames = [load_matrix(path) for path in frame_paths]
    ids = sample_indices(len(x), min(6, len(x)))
    sampled_x = [x[i] for i in ids]
    sampled_frames = [[frame[i] for i in ids] for frame in frames]

    payload = {
        "v": v,
        "times": times,
        "sampled_x": sampled_x,
        "frames": sampled_frames,
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>distribution_f animation</title>
  <style>
    :root {{
      --bg: #fbfaf7;
      --fg: #171717;
      --grid: #d8d2c7;
    }}
    body {{
      margin: 0;
      font-family: Georgia, serif;
      background: var(--bg);
      color: var(--fg);
    }}
    .wrap {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 600;
    }}
    .meta {{
      margin: 0 0 18px;
      font-size: 14px;
      color: #555;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background: white;
      border: 1px solid #ddd6ca;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    button {{
      padding: 8px 12px;
      border: 1px solid #bbb2a4;
      background: #f5f0e7;
      cursor: pointer;
    }}
    input[type="range"] {{
      flex: 1;
      min-width: 220px;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
      font-size: 13px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Distribution f(c) at Selected x</h1>
    <p class="meta">Browser animation fallback for environments where GIF preview fails.</p>
    <svg id="plot" viewBox="0 0 960 560" aria-label="distribution animation"></svg>
    <div class="controls">
      <button id="toggle">Pause</button>
      <label>Frame <span id="frameLabel">0</span></label>
      <input id="slider" type="range" min="0" max="0" value="0">
      <label>t = <span id="timeLabel">0.0000</span></label>
    </div>
    <div id="legend" class="legend"></div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const svg = document.getElementById('plot');
    const slider = document.getElementById('slider');
    const frameLabel = document.getElementById('frameLabel');
    const timeLabel = document.getElementById('timeLabel');
    const toggle = document.getElementById('toggle');
    const legend = document.getElementById('legend');
    const colors = ['#0d0887', '#5b02a3', '#9a179b', '#cb4679', '#ed7953', '#fdb42f'];
    const margin = {{ left: 74, right: 24, top: 28, bottom: 54 }};
    const W = 960, H = 560;
    const plotW = W - margin.left - margin.right;
    const plotH = H - margin.top - margin.bottom;
    const v = payload.v;
    const frames = payload.frames;
    const times = payload.times;
    const sampledX = payload.sampled_x;
    slider.max = String(frames.length - 1);

    const vMin = Math.min(...v);
    const vMax = Math.max(...v);
    const globalFMax = Math.max(...frames.flat(2));

    function sx(val) {{
      return margin.left + (val - vMin) / (vMax - vMin) * plotW;
    }}

    function sy(val, fMax) {{
      return H - margin.bottom - (val / fMax) * plotH;
    }}

    function line(x1, y1, x2, y2, color, width, dash='') {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      el.setAttribute('x1', x1);
      el.setAttribute('y1', y1);
      el.setAttribute('x2', x2);
      el.setAttribute('y2', y2);
      el.setAttribute('stroke', color);
      el.setAttribute('stroke-width', width);
      if (dash) el.setAttribute('stroke-dasharray', dash);
      svg.appendChild(el);
    }}

    function text(x, y, value, anchor='middle', size='13') {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      el.setAttribute('x', x);
      el.setAttribute('y', y);
      el.setAttribute('fill', '#222');
      el.setAttribute('font-size', size);
      el.setAttribute('text-anchor', anchor);
      el.textContent = value;
      svg.appendChild(el);
    }}

    function pathFor(profile, fMax) {{
      let d = '';
      for (let i = 0; i < v.length; i++) {{
        const xVal = sx(v[i]);
        const yVal = sy(profile[i], fMax);
        d += (i === 0 ? 'M' : 'L') + xVal + ' ' + yVal + ' ';
      }}
      return d;
    }}

    function render(frameIndex) {{
      svg.innerHTML = '';
      const frame = frames[frameIndex];
      const frameMax = Math.max(...frame.flat());
      const fMax = Math.max(frameMax * 1.05, globalFMax * 0.25);

      line(margin.left, H - margin.bottom, W - margin.right, H - margin.bottom, '#111', 1.5);
      line(margin.left, margin.top, margin.left, H - margin.bottom, '#111', 1.5);

      for (let i = 0; i <= 5; i++) {{
        const fx = margin.left + i / 5 * plotW;
        const fy = H - margin.bottom - i / 5 * plotH;
        line(fx, margin.top, fx, H - margin.bottom, '#d8d2c7', 1, '3 5');
        line(margin.left, fy, W - margin.right, fy, '#d8d2c7', 1, '3 5');
        const vTick = vMin + i / 5 * (vMax - vMin);
        const fTick = i / 5 * fMax;
        text(fx, H - margin.bottom + 22, vTick.toFixed(1), 'middle', '12');
        text(margin.left - 10, fy + 4, fTick.toFixed(2), 'end', '12');
      }}

      text(W / 2, H - 10, 'velocity c', 'middle', '16');
      text(22, H / 2, 'distribution f', 'middle', '16');
      text(W / 2, 18, `Distribution f(c) at Selected x, t=${{times[frameIndex].toFixed(4)}}`, 'middle', '18');

      frame.forEach((profile, i) => {{
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', pathFor(profile, fMax));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', colors[i % colors.length]);
        path.setAttribute('stroke-width', '2.4');
        svg.appendChild(path);
      }});

      frameLabel.textContent = String(frameIndex);
      timeLabel.textContent = times[frameIndex].toFixed(4);
      slider.value = String(frameIndex);
    }}

    legend.innerHTML = sampledX.map((xVal, i) =>
      `<span class="chip"><span class="swatch" style="background:${{colors[i % colors.length]}}"></span>x=${{xVal.toFixed(2)}}</span>`
    ).join('');

    let frame = 0;
    let playing = true;
    let timer = null;

    function start() {{
      if (timer) clearInterval(timer);
      timer = setInterval(() => {{
        frame = (frame + 1) % frames.length;
        render(frame);
      }}, 260);
    }}

    toggle.addEventListener('click', () => {{
      playing = !playing;
      toggle.textContent = playing ? 'Pause' : 'Play';
      if (playing) start();
      else if (timer) clearInterval(timer);
    }});

    slider.addEventListener('input', () => {{
      frame = Number(slider.value);
      render(frame);
    }});

    render(0);
    start();
  </script>
</body>
</html>
"""

    out = output_dir / "distribution_f.html"
    out.write_text(html, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
