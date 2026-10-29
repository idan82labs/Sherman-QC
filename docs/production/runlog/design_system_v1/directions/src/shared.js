// Shared data (real 10839000 FULL scan, run 20261019T-real-v2) and a small SVG part renderer.
window.SQ = (() => {
  // verdict: FIX | CHECK | OK ; cells: [expected, actual, delta, state] state: ok|check|fix|nm
  const bends = [
    { id: 'B1', seq: 1, v: 'CHECK', move: 1, a: ['3.6', null, null, 'nm'], b: ['13.0', '13.0', '−0.1', 'check'], g: ['145.0', '143.3', '−1.7', 'check'] },
    { id: 'B2', seq: 2, v: 'CHECK', move: 1, a: ['21.4', '21.4', '0.0', 'check'], b: ['3.6', null, null, 'nm'], g: ['145.0', '145.5', '+0.5', 'check'] },
    { id: 'B3', seq: 3, v: 'CHECK', a: ['3.6', null, null, 'nm'], b: ['13.0', '13.4', '+0.3', 'check'], g: ['145.0', '144.1', '−0.9', 'check'] },
    { id: 'B4', seq: 4, v: 'CHECK', a: ['21.4', '21.0', '−0.4', 'check'], b: ['3.6', null, null, 'nm'], g: ['145.0', '146.1', '+1.1', 'check'] },
    { id: 'B5', seq: 5, v: 'FIX', a: ['12.0', '12.0', '0.0', 'ok'], b: ['41.2', '41.3', '+0.1', 'ok'], g: ['90.0', '88.6', '−1.4', 'fix'] },
    { id: 'B10', seq: 6, v: 'CHECK', a: ['41.2', '41.3', '+0.1', 'ok'], b: ['312.9', '313.2', '+0.3', 'ok'], g: ['90.0', '88.8', '−1.2', 'check'] },
    { id: 'B7', seq: 7, v: 'FIX', a: ['12.0', '12.0', '0.0', 'ok'], b: ['41.2', '41.1', '−0.1', 'ok'], g: ['90.0', '88.5', '−1.5', 'fix'] },
    { id: 'B12', seq: 8, v: 'FIX', a: ['41.2', '41.2', '0.0', 'ok'], b: ['312.9', '313.2', '+0.3', 'ok'], g: ['90.0', '88.3', '−1.7', 'fix'] },
    { id: 'B6', seq: 9, v: 'CHECK', a: ['21.4', '21.0', '−0.4', 'check'], b: ['43.0', '42.9', '−0.1', 'ok'], g: ['90.0', '89.6', '−0.4', 'check'] },
    { id: 'B11', seq: 10, v: 'FIX', a: ['43.0', '43.0', '0.0', 'ok'], b: ['206.1', '206.3', '+0.2', 'ok'], g: ['90.0', '88.0', '−2.0', 'fix'] },
    { id: 'B8', seq: 11, v: 'CHECK', a: ['21.4', '21.3', '−0.1', 'check'], b: ['43.0', '43.0', '0.0', 'ok'], g: ['90.0', '89.0', '−1.0', 'check'] },
    { id: 'B9', seq: 12, v: 'CHECK', a: ['43.0', '43.1', '+0.1', 'ok'], b: ['206.1', '206.3', '+0.2', 'ok'], g: ['90.0', '88.5', '−1.5', 'check'] },
  ]
  const he = {
    FIX: 'לתקן', CHECK: 'לבדוק', OK: 'תקין', nm: 'לא נמדד',
    bend: 'כיפוף', result: 'תוצאה', fa: 'כנף A', fb: 'כנף B', ang: 'זווית',
    action: 'מחוץ לטולרנס. לתקן, או להפריש את החלק ולעדכן את ראש המשמרת.',
  }

  // ---------- geometry (mm). Frame plate z=0, outer flanges toward viewer (z<0), window lips away (z>0).
  const W = 206, H = 313, wx0 = 43, wx1 = 163, wy0 = 60, wy1 = 253, F = 41
  const j = 3.6, js = Math.sin(35 * Math.PI / 180) * j, jc = Math.cos(35 * Math.PI / 180) * j
  const P = (x, y, z) => [x, y, z]
  const quads = [
    // plate
    [P(0, 0, 0), P(W, 0, 0), P(W, wy0, 0), P(0, wy0, 0)],
    [P(0, wy1, 0), P(W, wy1, 0), P(W, H, 0), P(0, H, 0)],
    [P(0, wy0, 0), P(wx0, wy0, 0), P(wx0, wy1, 0), P(0, wy1, 0)],
    [P(wx1, wy0, 0), P(W, wy0, 0), P(W, wy1, 0), P(wx1, wy1, 0)],
    // outer flanges
    [P(0, 6, 0), P(0, H - 6, 0), P(0, H - 6, -F), P(0, 6, -F)],
    [P(W, 6, 0), P(W, H - 6, 0), P(W, H - 6, -F), P(W, 6, -F)],
    [P(6, 0, 0), P(W - 6, 0, 0), P(W - 6, 0, -F), P(6, 0, -F)],
    [P(6, H, 0), P(W - 6, H, 0), P(W - 6, H, -F), P(6, H, -F)],
    // joggles: top (B2 at tip, B1 after the 35° lip), bottom (B4, B3)
    [P(6, H, -F), P(W - 6, H, -F), P(W - 6, H + js, -F - jc), P(6, H + js, -F - jc)],
    [P(6, H + js, -F - jc), P(W - 6, H + js, -F - jc), P(W - 6, H + js, -F - jc - 13), P(6, H + js, -F - jc - 13)],
    [P(6, 0, -F), P(W - 6, 0, -F), P(W - 6, -js, -F - jc), P(6, -js, -F - jc)],
    [P(6, -js, -F - jc), P(W - 6, -js, -F - jc), P(W - 6, -js, -F - jc - 13), P(6, -js, -F - jc - 13)],
    // window lips (away from viewer)
    [P(wx0, wy0, 0), P(wx0, wy1, 0), P(wx0, wy1, 21), P(wx0, wy0, 21)],
    [P(wx1, wy0, 0), P(wx1, wy1, 0), P(wx1, wy1, 21), P(wx1, wy0, 21)],
    [P(wx0, wy0, 0), P(wx1, wy0, 0), P(wx1, wy0, 43), P(wx0, wy0, 43)],
    [P(wx0, wy1, 0), P(wx1, wy1, 0), P(wx1, wy1, 43), P(wx0, wy1, 43)],
  ]
  const lines = {
    B5: [P(0, 6, 0), P(0, H - 6, 0)], B7: [P(W, 6, 0), P(W, H - 6, 0)],
    B10: [P(6, 0, 0), P(W - 6, 0, 0)], B12: [P(6, H, 0), P(W - 6, H, 0)],
    B2: [P(6, H, -F), P(W - 6, H, -F)], B1: [P(6, H + js, -F - jc), P(W - 6, H + js, -F - jc)],
    B4: [P(6, 0, -F), P(W - 6, 0, -F)], B3: [P(6, -js, -F - jc), P(W - 6, -js, -F - jc)],
    B6: [P(wx0, wy0, 0), P(wx0, wy1, 0)], B8: [P(wx1, wy0, 0), P(wx1, wy1, 0)],
    B11: [P(wx0, wy0, 0), P(wx1, wy0, 0)], B9: [P(wx0, wy1, 0), P(wx1, wy1, 0)],
  }

  function projector({ w, h, scale, yaw = -32, pitch = 22, cx = W / 2, cy = H / 2, cz = 0, ox = 0, oy = 0 }) {
    const ay = yaw * Math.PI / 180, ap = pitch * Math.PI / 180
    return ([x, y, z]) => {
      x -= cx; y -= cy; z -= cz
      const x1 = x * Math.cos(ay) + z * Math.sin(ay), z1 = -x * Math.sin(ay) + z * Math.cos(ay)
      const y2 = y * Math.cos(ap) - z1 * Math.sin(ap), z2 = y * Math.sin(ap) + z1 * Math.cos(ap)
      return [w / 2 + ox + scale * x1, h / 2 + oy - scale * y2, z2]
    }
  }

  // opts: w,h,scale,yaw,pitch, face(l)->color, edge, edgeW, bendColor(b)->color|null, bendW,
  //       selected, selColor, glow, marker(b, x, y, sel)->svg, markerFor(b)->bool, markerBox [w,h], tick, extraDefs
  function render(opts) {
    const pr = projector(opts)
    const polys = quads.map(q => {
      const pts = q.map(pr)
      const d = pts.reduce((s, p) => s + p[2], 0) / 4
      // normal in screen space for shading
      const [a, b, c] = pts
      const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2]
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
      const n = Math.hypot(nx, ny, nz) || 1
      const lit = Math.abs((nx * -0.3 + ny * -0.6 + nz * -0.74) / n)
      return { pts, d, lit }
    }).sort((p, q) => q.d - p.d)
    let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${opts.w}" height="${opts.h}" viewBox="0 0 ${opts.w} ${opts.h}">`
    s += `<defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="${opts.glowBlur || 6}"/></filter>${opts.extraDefs || ''}</defs>`
    for (const p of polys) {
      s += `<polygon points="${p.pts.map(q => q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' ')}" fill="${opts.face(p.lit)}" stroke="${opts.edge}" stroke-width="${opts.edgeW || 1}" stroke-linejoin="round"/>`
      if (opts.faceOverlay) s += `<polygon points="${p.pts.map(q => q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' ')}" fill="${opts.faceOverlay}" stroke="none"/>`
    }
    const scr = {}
    for (const b of bends) {
      const [p0, p1] = lines[b.id].map(pr)
      scr[b.id] = { p0, p1, mid: [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2] }
      if (b.id === opts.selected) continue
      const col = opts.bendColor(b)
      if (!col) continue
      s += `<line x1="${p0[0]}" y1="${p0[1]}" x2="${p1[0]}" y2="${p1[1]}" stroke="${col}" stroke-width="${opts.bendW || 2}" stroke-linecap="round" ${opts.bendDash && b.v !== 'FIX' ? `stroke-dasharray="${opts.bendDash}"` : ''}/>`
    }
    if (opts.selected) {
      const { p0, p1 } = scr[opts.selected]
      if (opts.glow) s += `<line x1="${p0[0]}" y1="${p0[1]}" x2="${p1[0]}" y2="${p1[1]}" stroke="${opts.glow}" stroke-width="${opts.glowW || 16}" stroke-linecap="round" filter="url(#glow)" opacity=".9"/>`
      s += `<line x1="${p0[0]}" y1="${p0[1]}" x2="${p1[0]}" y2="${p1[1]}" stroke="${opts.selColor}" stroke-width="${opts.selW || 5}" stroke-linecap="round"/>`
      if (opts.selCore) s += `<line x1="${p0[0]}" y1="${p0[1]}" x2="${p1[0]}" y2="${p1[1]}" stroke="${opts.selCore}" stroke-width="1.5" stroke-linecap="round"/>`
    }
    // markers with simple collision avoidance: slide along the bend line, then off it (short tick)
    const placed = []
    const [mw, mh] = opts.markerBox || [34, 24]
    const order = bends.filter(b => opts.markerFor(b)).sort((a, b) => (b.id === opts.selected) - (a.id === opts.selected))
    for (const b of order) {
      const { p0, p1 } = scr[b.id]
      const dx = p1[0] - p0[0], dy = p1[1] - p0[1], L = Math.hypot(dx, dy) || 1
      const nx = -dy / L, ny = dx / L
      let best = null
      outer: for (const off of [0, 22, -22, 34, -34]) {
        for (const t of [0.5, 0.35, 0.65, 0.22, 0.78, 0.12, 0.88]) {
          const x = p0[0] + dx * t + nx * off, y = p0[1] + dy * t + ny * off
          const box = [x - mw / 2 - 3, y - mh / 2 - 3, x + mw / 2 + 3, y + mh / 2 + 3]
          if (placed.every(q => box[2] < q[0] || box[0] > q[2] || box[3] < q[1] || box[1] > q[3])) {
            best = { x, y, ax: p0[0] + dx * t, ay: p0[1] + dy * t, off }
            placed.push(box)
            break outer
          }
        }
      }
      if (!best) continue
      if (best.off !== 0) s += `<line x1="${best.ax}" y1="${best.ay}" x2="${best.x}" y2="${best.y}" stroke="${opts.tick || '#888'}" stroke-width="1"/><circle cx="${best.ax}" cy="${best.ay}" r="2.2" fill="${opts.tick || '#888'}"/>`
      s += opts.marker(b, best.x, best.y, b.id === opts.selected)
    }
    return s + '</svg>'
  }
  const fmt = c => c[1] == null ? null : c
  return { bends, he, render, fmt, byId: id => bends.find(b => b.id === id) }
})()
