const waveform = document.getElementById('sound-waveform');
for (let index = 0; index < 110; index++) {
  const bar = document.createElement('span');
  const pulse = Math.abs(Math.sin(index * .24) * Math.cos(index * .073));
  const noise = Math.abs(Math.sin(index * 2.17));
  bar.style.height = `${9 + Math.round((pulse * .77 + noise * .23) * 92)}%`;
  waveform.append(bar);
}

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
// The home visual now lives in terrain-bg.js; these stay as hooks for the code below.
function resizeMesh() {}
function queueFrame() {}
function updateMeshFromScroll() {}
window.addEventListener('resize', resizeMesh, { passive: true });
const horizontalTrack = document.querySelector('main');
horizontalTrack.addEventListener('scroll', updateMeshFromScroll, { passive: true });

const tabLinks = document.querySelectorAll('.section-tabs a');
function markCurrentPanel() {
  const panel = horizontalTrack.children[Math.round(horizontalTrack.scrollLeft / Math.max(1, horizontalTrack.clientWidth))];
  tabLinks.forEach(link => link.setAttribute('aria-current', String(Boolean(panel) && link.hash === `#${panel.id}`)));
}
horizontalTrack.addEventListener('scroll', markCurrentPanel, { passive: true });
markCurrentPanel();

// Glide between panels with our own animation: a native smooth scroll can be
// cut short, and snapping would fight each step, so it is paused meanwhile.
let glideFrame = 0;
function glideTo(left) {
  cancelAnimationFrame(glideFrame);
  const start = horizontalTrack.scrollLeft, distance = left - start;
  if (Math.abs(distance) < 1) return;
  if (prefersReducedMotion) { horizontalTrack.scrollLeft = left; return; }
  horizontalTrack.style.scrollSnapType = 'none';
  const startedAt = performance.now();
  const step = now => {
    const progress = Math.min(1, (now - startedAt) / 450);
    horizontalTrack.scrollLeft = start + distance * (1 - Math.pow(1 - progress, 3));
    if (progress < 1) { glideFrame = requestAnimationFrame(step); return; }
    glideFrame = 0;
    horizontalTrack.style.scrollSnapType = '';
  };
  glideFrame = requestAnimationFrame(step);
}

// Settle on the nearest panel whenever a scroll stops between two of them.
let settleTimer;
horizontalTrack.addEventListener('scroll', () => {
  clearTimeout(settleTimer);
  settleTimer = setTimeout(() => {
    if (glideFrame) return;
    const width = horizontalTrack.clientWidth;
    glideTo(Math.round(horizontalTrack.scrollLeft / width) * width);
  }, 160);
}, { passive: true });

// The wheel turns one gesture into one panel step; nudging scrollLeft a few
// pixels at a time gets pulled back by scroll snapping.
let wheelTotal = 0, wheelLastAt = 0, wheelLockedUntil = 0;
horizontalTrack.addEventListener('wheel', event => {
  if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
  for (let node = event.target instanceof Element ? event.target : null; node && node !== horizontalTrack; node = node.parentElement) {
    const style = getComputedStyle(node);
    const canScroll = /(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 8;
    if (canScroll) {
      const canMove = event.deltaY < 0 ? node.scrollTop > 0 : node.scrollTop < node.scrollHeight - node.clientHeight - 1;
      if (canMove) return;
    }
  }
  event.preventDefault();
  const now = performance.now();
  if (now < wheelLockedUntil) {
    // Swallow trackpad momentum until the gesture goes quiet.
    wheelLockedUntil = Math.max(wheelLockedUntil, now + 180);
    return;
  }
  if (now - wheelLastAt > 250) wheelTotal = 0;
  wheelLastAt = now;
  wheelTotal += event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? horizontalTrack.clientHeight : 1);
  if (Math.abs(wheelTotal) < 40) return;
  const width = horizontalTrack.clientWidth;
  const current = Math.round(horizontalTrack.scrollLeft / width);
  const next = Math.max(0, Math.min(horizontalTrack.children.length - 1, current + Math.sign(wheelTotal)));
  wheelTotal = 0;
  wheelLockedUntil = now + 550;
  if (next !== current) glideTo(next * width);
}, { passive: false });

// Project cards open their write-up in the blog reader; Back returns here.
document.querySelectorAll('a[data-return]').forEach(link => {
  link.addEventListener('click', () => { window.blogReturnHash = link.dataset.return; });
});

document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', event => {
    const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
    if (!target || !(target === horizontalTrack || horizontalTrack.contains(target))) return;
    event.preventDefault();
    const trackLeft = horizontalTrack.getBoundingClientRect().left;
    const left = target === horizontalTrack ? 0 : target.getBoundingClientRect().left - trackLeft + horizontalTrack.scrollLeft;
    horizontalTrack.scrollTo({ left, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
    if (window.location.hash !== link.hash) window.location.hash = link.hash;
  });
});

window.addEventListener('hashchange', () => {
  if (!window.location.hash.startsWith('#/blog')) return;
  const blogs = document.getElementById('blogs');
  const trackLeft = horizontalTrack.getBoundingClientRect().left;
  horizontalTrack.scrollTo({ left: blogs.getBoundingClientRect().left - trackLeft + horizontalTrack.scrollLeft, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
});

document.addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight'].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey) return;
  const target = event.target;
  if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
  event.preventDefault();
  horizontalTrack.scrollBy({ left: event.key === 'ArrowRight' ? horizontalTrack.clientWidth : -horizontalTrack.clientWidth, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
});

const themeToggle = document.getElementById('theme-toggle');
let currentTheme = 'light';
// Every visit opens in light mode; the choice only lasts for the current tab.
try { currentTheme = sessionStorage.getItem('pt-site-theme') || 'light'; } catch { /* Local file previews may disable storage. */ }
function applyTheme(theme) {
  currentTheme = theme;
  document.documentElement.dataset.theme = theme;
  themeToggle.setAttribute('aria-checked', String(theme === 'dark'));
  window.dispatchEvent(new CustomEvent('site-theme-change', { detail: theme }));
  queueFrame();
  try { sessionStorage.setItem('pt-site-theme', theme); } catch { /* Keep the toggle usable without storage. */ }
}
applyTheme(currentTheme);
themeToggle.addEventListener('click', () => applyTheme(currentTheme === 'dark' ? 'light' : 'dark'));

if (window.location.hash.startsWith('#/blog')) {
  const blogsPanel = document.getElementById('blogs');
  window.setTimeout(() => {
    const trackLeft = horizontalTrack.getBoundingClientRect().left;
    horizontalTrack.scrollLeft = blogsPanel.getBoundingClientRect().left - trackLeft + horizontalTrack.scrollLeft;
  }, 0);
}

resizeMesh(); updateMeshFromScroll();
