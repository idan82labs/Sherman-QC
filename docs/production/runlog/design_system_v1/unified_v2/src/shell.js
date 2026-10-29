// AppShell for office screens (light): sidebar with the Sherman logo baked in.
window.shell = (active) => {
  const nav = [['dashboard', 'לוח בקרה'], ['parts', 'חלקים לייצור'], ['station', 'עמדת בדיקה'], ['quarantine', 'הסגר']]
  const adm = [['user', 'משתמשים'], ['settings', 'הגדרות']]
  const a = ([i, t]) => `<a class="${t === active ? 'on' : ''}">${icon(i, 20)}<span>${t}</span></a>`
  return `<nav class="side"><div class="logo"><img src="sherman-logo-colour.png" alt="Sherman"></div>
  <div class="sec">ייצור</div>${nav.map(a).join('')}<div class="sec">ניהול</div>${adm.map(a).join('')}
  <div class="user"><span class="av">EA</span><div><div>e2e-admin</div><div class="lbl">מנהל</div></div><span style="margin-inline-start:auto" class="badge">English</span></div></nav>`
}
document.write(`<style>
.side{background:var(--surface);border-inline-end:1px solid var(--line);display:flex;flex-direction:column;width:248px}
.side .logo{height:72px;display:flex;align-items:center;padding:0 22px;border-bottom:1px solid var(--line)}
.side .logo img{height:32px}
.side .sec{font-size:var(--t-xs);color:var(--tx-3);padding:20px 22px 6px}
.side a{display:flex;align-items:center;gap:12px;height:44px;padding:0 22px;color:var(--tx-2);font-size:var(--t-md)}
.side a.on{color:var(--accent);background:var(--accent-bg);box-shadow:inset -3px 0 0 var(--accent);font-weight:600}
.side .user{margin-top:auto;border-top:1px solid var(--line);padding:14px 22px;display:flex;gap:10px;align-items:center;font-size:var(--t-sm)}
.side .av{width:32px;height:32px;border-radius:var(--r-1);background:var(--sh-navy);color:#fff;display:grid;place-items:center;font-size:12px;font-weight:700}
</style>`)
