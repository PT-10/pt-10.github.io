// Loaded as a classic script so the blog also works when index.html is opened
// straight from disk, where browsers refuse to load local module files.
(async () => {
const [
  { default: React },
  { createRoot },
  { default: ReactMarkdown },
  { default: remarkMath },
  { default: rehypeKatex },
  { default: remarkGfm },
  { default: rehypeSlug },
  { default: rehypeRaw },
  { Prism: SyntaxHighlighter },
  { oneDark, oneLight },
] = await Promise.all([
  import('react'),
  import('react-dom/client'),
  import('react-markdown'),
  import('remark-math'),
  import('rehype-katex'),
  import('remark-gfm'),
  import('rehype-slug'),
  import('rehype-raw'),
  import('react-syntax-highlighter'),
  import('react-syntax-highlighter/dist/esm/styles/prism'),
]);
const { useEffect, useMemo, useRef, useState } = React;
const posts = window.BLOG_POSTS || [];

const h = React.createElement;
const equationLabelPattern = /\\label\{([^{}]+)\}/g;
const equationRefPattern = /\\(eqref|ref)\{([^{}]+)\}/g;

const toEquationId = label => `eq-${label.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-')}`;
const addEquationTag = (latex, number) => /\\tag\{[^}]*\}/.test(latex) ? latex : `${latex} \\tag{${number}}`;

const createEquationRefPlugin = () => tree => {
  const labels = new Map();
  let counter = 0;
  const registerLabels = node => {
    if (!node || !Array.isArray(node.children)) return;
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      if (child?.type === 'math' && typeof child.value === 'string') {
        const matches = [...child.value.matchAll(equationLabelPattern)];
        if (matches.length) {
          const label = matches[matches.length - 1][1]?.trim();
          if (label) {
            let entry = labels.get(label);
            if (!entry) { counter += 1; entry = { id: toEquationId(label), number: counter }; labels.set(label, entry); }
            child.value = addEquationTag(child.value.replace(equationLabelPattern, '').trim(), entry.number);
            const previous = node.children[i - 1];
            if (!(previous?.type === 'html' && previous.value?.includes(`id="${entry.id}"`))) {
              node.children.splice(i, 0, { type: 'html', value: `<span id="${entry.id}" class="eq-anchor" aria-hidden="true"></span>` });
              i += 1;
            }
          }
        }
      }
      registerLabels(child);
    }
  };
  const replaceRefs = (node, parentType) => {
    if (!node || !Array.isArray(node.children)) return;
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      if (child?.type === 'text' && typeof child.value === 'string' && parentType !== 'link') {
        const value = child.value;
        equationRefPattern.lastIndex = 0;
        const replacements = [];
        let start = 0, match;
        while ((match = equationRefPattern.exec(value)) !== null) {
          if (match.index > start) replacements.push({ type: 'text', value: value.slice(start, match.index) });
          const entry = labels.get(match[2].trim());
          replacements.push(entry ? { type: 'link', url: `#${entry.id}`, children: [{ type: 'text', value: match[1] === 'eqref' ? `(${entry.number})` : `${entry.number}` }] } : { type: 'text', value: match[0] });
          start = match.index + match[0].length;
        }
        if (start) {
          if (start < value.length) replacements.push({ type: 'text', value: value.slice(start) });
          node.children.splice(i, 1, ...replacements);
          i += replacements.length - 1;
          continue;
        }
      }
      replaceRefs(child, child?.type);
    }
  };
  registerLabels(tree);
  replaceRefs(tree);
};

const remarkPlugins = [remarkMath, createEquationRefPlugin, remarkGfm];
const rehypePlugins = [rehypeRaw, rehypeKatex, rehypeSlug];

function routeFromHash() {
  const match = window.location.hash.match(/^#\/blog(?:\/([^#]+))?(?:#(.+))?$/);
  return match ? { id: match[1] || null, anchor: match[2] || null } : { id: null, anchor: null };
}

function imagePath(src) {
  if (!src || /^(https?:|data:|blob:)/i.test(src)) return src;
  const clean = decodeURIComponent(src).replace(/^\/+/, '');
  const path = clean.startsWith('blog/') ? clean : `blog/${clean}`;
  return path.split('/').map(encodeURIComponent).join('/');
}

function App() {
  const initialRoute = routeFromHash();
  const [selectedPost, setSelectedPost] = useState(() => posts.find(post => post.id === initialRoute.id) || null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTag, setActiveTag] = useState(null);
  const [toc, setToc] = useState([]);
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || 'light');
  const articleRef = useRef(null);
  const previousPosts = useRef([]);

  const listedPosts = useMemo(() => posts.filter(post => !post.hidden), []);
  const allTags = useMemo(() => [...new Set(listedPosts.flatMap(post => post.tags || []))].sort(), [listedPosts]);
  const filteredPosts = useMemo(() => listedPosts.filter(post => {
    if (activeTag && !post.tags.includes(activeTag)) return false;
    if (!searchQuery.trim()) return true;
    const haystack = `${post.title} ${post.snippet} ${(post.tags || []).join(' ')}`.toLowerCase();
    return searchQuery.trim().toLowerCase().split(/\s+/).every(token => haystack.includes(token));
  }), [activeTag, searchQuery]);

  function selectPost(id, anchor = null) {
    const next = posts.find(post => post.id === id) || null;
    if (!next) return;
    if (selectedPost && selectedPost.id !== next.id) previousPosts.current.push(selectedPost.id);
    setSelectedPost(next);
    window.location.hash = `/blog/${next.id}${anchor ? `#${anchor}` : ''}`;
    if (anchor) window.setTimeout(() => document.getElementById(anchor)?.scrollIntoView({ behavior: 'smooth' }), 120);
  }

  function goBack() {
    const previousId = previousPosts.current.pop();
    if (previousId) {
      const previous = posts.find(post => post.id === previousId) || null;
      setSelectedPost(previous);
      window.location.hash = `/blog/${previousId}`;
    }
    else {
      // Posts opened from a project card return to the projects panel.
      const returnTo = window.blogReturnHash || 'blogs';
      window.blogReturnHash = null;
      setSelectedPost(null);
      setToc([]);
      window.location.hash = returnTo;
    }
    articleRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }

  useEffect(() => {
    const onHashChange = () => {
      const route = routeFromHash();
      const next = posts.find(post => post.id === route.id) || null;
      setSelectedPost(next);
      if (!next) previousPosts.current = [];
      if (route.anchor) window.setTimeout(() => document.getElementById(route.anchor)?.scrollIntoView({ behavior: 'smooth' }), 140);
    };
    window.addEventListener('hashchange', onHashChange);
    const onThemeChange = event => setTheme(event.detail || document.documentElement.dataset.theme || 'light');
    window.addEventListener('site-theme-change', onThemeChange);
    return () => {
      window.removeEventListener('hashchange', onHashChange);
      window.removeEventListener('site-theme-change', onThemeChange);
    };
  }, []);

  useEffect(() => {
    const parent = document.getElementById('blogs');
    parent?.classList.toggle('is-reading', Boolean(selectedPost));
    if (!selectedPost) { setToc([]); return; }
    const headings = [];
    let inFence = false;
    selectedPost.content.split('\n').forEach(line => {
      if (/^\s*(```|~~~)/.test(line)) { inFence = !inFence; return; }
      if (inFence) return;
      const match = line.match(/^\s*(#{1,3})\s+(.*)$/);
      if (!match) return;
      const text = match[2].replace(/`([^`]+)`/g, '$1').replace(/\$[^$]*\$/g, '').trim();
      const id = text.toLowerCase().trim().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
      headings.push({ id, text, level: match[1].length });
    });
    setToc(headings);
  }, [selectedPost]);

  const CodeBlock = ({ node, inline, className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className || '');
    return !inline && match
      ? h(SyntaxHighlighter, { style: theme === 'dark' ? oneDark : oneLight, language: match[1], PreTag: 'div', customStyle: { margin: 0, borderRadius: '0.45rem', fontSize: '0.875rem', lineHeight: '1.45', padding: '1rem', background: theme === 'dark' ? '#171a18' : '#e9e8e1' }, ...props }, String(children).trim())
      : h('code', { className, ...props }, children);
  };

  const ImageRenderer = ({ node, src, alt, title, ...props }) => {
    const [currentSrc, setCurrentSrc] = useState(() => imagePath(src));
    useEffect(() => {
      const localSrc = imagePath(src);
      setCurrentSrc(localSrc);
      if (theme !== 'dark' || !localSrc || /^(https?:|data:|blob:)/i.test(localSrc)) return;
      const dot = localSrc.lastIndexOf('.');
      if (dot < 0) return;
      const darkSrc = `${localSrc.slice(0, dot)}-dark${localSrc.slice(dot)}`;
      const image = new Image();
      image.onload = () => setCurrentSrc(darkSrc);
      image.src = darkSrc;
    }, [src, theme]);
    return h('img', { src: currentSrc, alt, title, className: 'blog-image', style: title ? { width: title } : undefined, ...props });
  };

  const MarkdownLink = ({ node, href = '', children, ...props }) => h('a', { href, ...props, onClick: event => {
    if (href.startsWith('#/blog/')) {
      event.preventDefault();
      const match = href.match(/^#\/blog\/([^#]+)(?:#(.+))?$/);
      if (match) selectPost(match[1], match[2] || null);
    } else if (href.startsWith('#')) {
      event.preventDefault();
      const id = decodeURIComponent(href.slice(1));
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
      if (selectedPost) history.replaceState(null, '', `#/blog/${selectedPost.id}#${id}`);
    }
  } }, children);

  const markdownComponents = { code: CodeBlock, img: ImageRenderer, a: MarkdownLink };

  if (selectedPost) {
    return h('div', { className: 'blog-reader' },
      h('aside', { className: 'blog-toc' },
        h('button', { type: 'button', className: 'blog-back', onClick: goBack }, ({ work: '← Back to projects', home: '← Back to home' })[window.blogReturnHash] || '← Back to blogs'),
        h('h3', null, 'On this page'),
        h('nav', null, toc.map(heading => h('a', { key: heading.id, href: `#${heading.id}`, className: `toc-level-${heading.level}`, onClick: event => { event.preventDefault(); document.getElementById(heading.id)?.scrollIntoView({ behavior: 'smooth' }); } }, heading.text)))
      ),
      h('article', { className: 'blog-article-scroll', ref: articleRef },
        h('div', { className: 'blog-article-inner' },
          h('button', { type: 'button', className: 'blog-back blog-back-mobile', onClick: goBack }, ({ work: '← Back to projects', home: '← Back to home' })[window.blogReturnHash] || '← Back to blogs'),
          selectedPost.banner ? h('img', { className: 'blog-banner', src: imagePath(selectedPost.banner), alt: '' }) : null,
          h('div', { className: 'blog-post-header' },
            h('span', { className: 'blog-date' }, selectedPost.date),
            h('h1', null, selectedPost.title),
            h('span', { className: 'blog-read-time' }, selectedPost.readTime)
          ),
          h('div', { className: 'blog-tags post-tags' }, (selectedPost.tags || []).map(tag => h('span', { key: tag }, tag))),
          h('div', { className: 'blog-prose' }, h(ReactMarkdown, { remarkPlugins, rehypePlugins, components: markdownComponents }, selectedPost.content))
        )
      )
    );
  }

  return h('div', { className: 'blog-list-view' },
    h('div', { className: 'blog-tools' },
      h('label', { className: 'blog-search' },
        h('span', { className: 'search-icon', 'aria-hidden': true }, '⌕'),
        h('input', { type: 'search', value: searchQuery, placeholder: 'Search blogs', 'aria-label': 'Search blogs', onChange: event => setSearchQuery(event.target.value) })
      ),
      h('div', { className: 'blog-filters', role: 'group', 'aria-label': 'Filter blogs by topic' },
        h('button', { type: 'button', className: activeTag === null ? 'active' : '', onClick: () => setActiveTag(null) }, 'All'),
        allTags.map(tag => h('button', { type: 'button', key: tag, className: activeTag === tag ? 'active' : '', onClick: () => setActiveTag(activeTag === tag ? null : tag) }, tag))
      )
    ),
    filteredPosts.length ? h('div', { className: 'blog-post-list' }, filteredPosts.map(post => h('button', { type: 'button', className: 'blog-post-card', key: post.id, onClick: () => selectPost(post.id) },
      h('div', { className: 'blog-card-meta' }, h('span', { className: 'blog-date' }, post.date), h('span', { className: 'blog-read-time' }, post.readTime)),
      h('div', { className: 'blog-card-body' },
        h('h3', { className: 'blog-title' }, post.title),
        h('p', { className: 'blog-snippet' }, h(ReactMarkdown, { components: { p: ({ children }) => h(React.Fragment, null, children), code: ({ children }) => h('code', { className: 'blog-inline-code' }, children) } }, post.snippet)),
        h('div', { className: 'blog-tags' }, (post.tags || []).map(tag => h('span', { key: tag }, tag)))),
      h('span', { className: 'blog-card-read', 'aria-hidden': 'true' }, 'Read ↗')
    ))) : h('p', { className: 'blog-empty' }, 'No blogs match your search.')
  );
}

const root = document.getElementById('blog-app');
if (root) createRoot(root).render(h(App));
})();
