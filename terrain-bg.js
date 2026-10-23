// Home background: a static, procedurally generated terrain. On load the mesh
// refines level by level (geomorphing like game terrain LOD): each new level's
// vertices start on the coarser surface and ease to their true height while
// their edges fade in, so the shape emerges smoothly instead of popping.
(() => {
  const canvas = document.getElementById('terrain-bg');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const levels = [4, 8, 16, 32, 64]; // columns; rows are 3/4 of these, so each level nests the previous one
  const stepMs = 650;
  let width = 0, height = 0, start = 0, frame = 0;

  const hash = (x, z) => { const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453; return s - Math.floor(s); };
  const smooth = t => t * t * (3 - 2 * t);
  function noise(x, z) {
    const xi = Math.floor(x), zi = Math.floor(z), xf = smooth(x - xi), zf = smooth(z - zi);
    const a = hash(xi, zi), b = hash(xi + 1, zi), c = hash(xi, zi + 1), d = hash(xi + 1, zi + 1);
    return a + (b - a) * xf + (c - a) * zf + (a - b - c + d) * xf * zf;
  }
  // Ridged mountains on the right and far side, low rolling ground near the text.
  function terrain(u, v) {
    let ridge = 0, amp = 1, freq = 2.2;
    for (let octave = 0; octave < 4; octave++) {
      const n = 1 - Math.abs(noise(u * freq + 3.1, v * freq + 7.7) * 2 - 1);
      ridge += n * n * amp; amp *= .5; freq *= 2.1;
    }
    const mountains = Math.max(0, Math.min(1, (u - .25) * 1.6)) * (.45 + v * .55);
    return ridge * .55 * mountains + noise(u * 3, v * 3) * .08;
  }

  // Heights per level, computed once.
  const heights = levels.map(cols => {
    const rows = cols * 3 / 4, grid = [];
    for (let r = 0; r <= rows; r++) {
      const row = [];
      for (let c = 0; c <= cols; c++) row.push(terrain(c / cols, r / rows));
      grid.push(row);
    }
    return grid;
  });
  function coarseHeight(level, c, r) {
    // Bilinear sample of the previous level at this vertex.
    const grid = heights[level - 1], x = c / 2, y = r / 2;
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, grid[0].length - 1), y1 = Math.min(y0 + 1, grid.length - 1);
    const fx = x - x0, fy = y - y0;
    const top = grid[y0][x0] * (1 - fx) + grid[y0][x1] * fx, bottom = grid[y1][x0] * (1 - fx) + grid[y1][x1] * fx;
    return top * (1 - fy) + bottom * fy;
  }

  function project(u, v, h) {
    const z = .55 + v * 2.6, x = (u - .5) * 3.4;
    const horizon = height * .3, focal = height * .62;
    return [width * .5 + x / z * focal * 1.15, horizon + (1.15 - h * 1.6) / z * focal];
  }

  function draw(level, t) {
    const dark = document.documentElement.dataset.theme === 'dark';
    ctx.clearRect(0, 0, width, height);

    const cols = levels[level], rows = cols * 3 / 4, ease = smooth(t);
    const grid = [];
    for (let r = 0; r <= rows; r++) {
      const row = [];
      for (let c = 0; c <= cols; c++) {
        const fresh = level > 0 && (c % 2 || r % 2);
        const target = heights[level][r][c];
        const h = fresh ? coarseHeight(level, c, r) + (target - coarseHeight(level, c, r)) * ease : target;
        row.push([...project(c / cols, r / rows, h), h, r / rows, fresh]);
      }
      grid.push(row);
    }

    // Batch edges into a few alpha buckets so each frame is a handful of strokes.
    const buckets = Array.from({ length: 12 }, () => new Path2D());
    const settled = .35 + .65 * (level + ease) / levels.length;
    const add = (a, b) => {
      let alpha = (1 - (a[3] + b[3]) / 2 * .75) * (.18 + (a[2] + b[2]) * .55) * settled;
      if (a[4] || b[4]) alpha *= ease;
      const index = Math.min(11, Math.floor(alpha * 14));
      if (index <= 0) return;
      buckets[index].moveTo(a[0], a[1]); buckets[index].lineTo(b[0], b[1]);
    };
    for (let r = 0; r <= rows; r++) for (let c = 0; c <= cols; c++) {
      const p = grid[r][c];
      if (c < cols) add(p, grid[r][c + 1]);
      if (r < rows) add(p, grid[r + 1][c]);
      if (r < rows && c < cols) add(p, grid[r + 1][c + 1]);
    }
    const tone = dark ? '170,224,198' : '96,170,138';
    ctx.lineWidth = level > 2 ? .7 : 1;
    buckets.forEach((path, index) => {
      if (!index) return;
      ctx.strokeStyle = `rgba(${tone},${((index / 14) * (dark ? .7 : .8)).toFixed(3)})`;
      ctx.stroke(path);
    });

    // Keep the text column clean: fade the terrain out toward the left and the top.
    const paper = getComputedStyle(document.documentElement).getPropertyValue('--paper').trim() || '#efeee9';
    const mask = document.createElement('canvas');
    mask.width = canvas.width; mask.height = canvas.height;
    const m = mask.getContext('2d');
    m.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
    const veil = m.createLinearGradient(0, 0, width * .62, 0);
    veil.addColorStop(0, paper); veil.addColorStop(.6, paper + 'e6'); veil.addColorStop(1, paper + '00');
    m.fillStyle = veil; m.fillRect(0, 0, width * .62, height);
    m.globalCompositeOperation = 'destination-in';
    const down = m.createLinearGradient(0, height * .7, 0, height * .95);
    down.addColorStop(0, '#000'); down.addColorStop(1, 'rgba(0,0,0,0)');
    m.fillStyle = down; m.fillRect(0, 0, width, height);
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(mask, 0, 0); ctx.restore();
    const top = ctx.createLinearGradient(0, 0, 0, height * .34);
    top.addColorStop(0, paper); top.addColorStop(1, paper + '00');
    ctx.fillStyle = top; ctx.fillRect(0, 0, width, height * .34);
  }

  function progress(now) {
    if (reduced || !start) return [levels.length - 1, 1];
    const elapsed = (now - start) / stepMs;
    if (elapsed >= levels.length) return [levels.length - 1, 1];
    return [Math.floor(elapsed), elapsed % 1];
  }
  function tick(now) {
    frame = 0;
    const [level, t] = progress(now);
    draw(level, t);
    if (level < levels.length - 1 || t < 1) frame = requestAnimationFrame(tick);
  }
  function resize() {
    const box = canvas.getBoundingClientRect();
    const ratio = Math.min(devicePixelRatio || 1, 2);
    width = box.width; height = box.height;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (!frame) draw(...progress(performance.now()));
  }
  addEventListener('resize', resize, { passive: true });
  addEventListener('site-theme-change', () => { if (!frame) requestAnimationFrame(now => draw(...progress(now))); });
  resize();
  if (!reduced) { start = performance.now() + 200; frame = requestAnimationFrame(tick); }
})();
