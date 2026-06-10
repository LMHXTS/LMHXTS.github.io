---
hide:
  - navigation
  - toc
---

<style>
  /* ===== Import Fonts ===== */
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700;900&family=JetBrains+Mono:ital,wght@0,400;0,500;0,700;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

  /* ===== CSS Variables (aligned with Material Indigo/Deep Purple) ===== */
  :root {
    --accent: #7c4dff;
    --accent-glow: rgba(124, 77, 255, 0.25);
    --surface: #111118;
    --surface-raised: #1a1a24;
    --surface-overlay: #1e1e2a;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --text-primary: #e8e6f0;
    --text-secondary: #9996a8;
    --text-tertiary: #6b6880;
    --terminal-green: #a6e22e;
    --terminal-blue: #5cc5ef;
    --terminal-red: #ff5f56;
    --terminal-yellow: #ffbd2e;
    --terminal-green-dim: #27c93f;
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --shadow-card: 0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.2);
    --shadow-card-hover: 0 4px 8px rgba(0,0,0,0.4), 0 12px 32px rgba(0,0,0,0.3);
    --transition-smooth: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* ===== Hero Section ===== */
  .hero-wrapper {
    position: relative;
    margin: 1.5rem 0 2.5rem;
  }

  /* Decorative glow behind terminal */
  .hero-wrapper::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 600px;
    height: 400px;
    background: radial-gradient(ellipse at center, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: pulse-glow 4s ease-in-out infinite alternate;
  }

  @keyframes pulse-glow {
    0% { opacity: 0.4; transform: translate(-50%, -50%) scale(0.95); }
    100% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.08); }
  }

  /* Terminal Window */
  .terminal-hero {
    position: relative;
    z-index: 1;
    background: linear-gradient(160deg, #1c1c28 0%, #16161f 40%, #1a1a26 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: clamp(1.2rem, 4vw, 2rem);
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    color: #e0e0e0;
    box-shadow:
      0 0 0 1px rgba(255,255,255,0.04) inset,
      var(--shadow-card-hover);
    transition: all var(--transition-smooth);
    word-break: break-word;
    white-space: normal;
  }

  /* Terminal Header Bar */
  .terminal-header {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }

  .terminal-title {
    margin-left: 0.5rem;
    font-size: 0.75em;
    color: var(--text-tertiary);
    letter-spacing: 0.05em;
    font-weight: 500;
  }

  .dot {
    width: 13px;
    height: 13px;
    border-radius: 50%;
    position: relative;
    transition: transform 0.15s ease;
  }

  .dot:hover { transform: scale(1.2); }

  .dot-red {
    background: radial-gradient(circle at 40% 35%, #ff6b6b, #cc3d3d 80%);
    box-shadow: 0 0 6px rgba(255, 95, 86, 0.5);
  }
  .dot-yellow {
    background: radial-gradient(circle at 40% 35%, #ffe06b, #cca83d 80%);
    box-shadow: 0 0 6px rgba(255, 189, 46, 0.5);
  }
  .dot-green {
    background: radial-gradient(circle at 40% 35%, #6be067, #3dcc46 80%);
    box-shadow: 0 0 6px rgba(39, 201, 63, 0.5);
  }

  /* Terminal Body */
  .terminal-body {
    font-size: clamp(0.85em, 2.5vw, 1em);
    line-height: 1.75;
  }

  .terminal-body .line {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    margin-bottom: 0.35rem;
  }

  .prompt {
    color: var(--terminal-blue);
    font-weight: 700;
    margin-right: 0.5rem;
    user-select: none;
    white-space: nowrap;
  }

  .command {
    color: #f0f0f2;
  }

  .text-output {
    color: #b0afba;
    font-style: italic;
    display: block;
    margin: 0.3rem 0 1rem 0;
    padding-left: 0.3rem;
    line-height: 1.5;
    border-left: 2px solid rgba(255,255,255,0.08);
  }

  .sys-output {
    color: var(--terminal-green);
    display: block;
    margin: 0.15rem 0;
    padding-left: 0.3rem;
    animation: fadeInLine 0.4s ease-out both;
  }
  .sys-output:nth-child(2) { animation-delay: 0.15s; }
  .sys-output:nth-child(3) { animation-delay: 0.3s; }
  .sys-output:nth-child(4) { animation-delay: 0.45s; }

  @keyframes fadeInLine {
    from { opacity: 0; transform: translateX(-8px); }
    to { opacity: 1; transform: translateX(0); }
  }

  /* Cursor blink */
  .cursor-blink {
    display: inline-block;
    width: 2px;
    height: 1.1em;
    background: var(--accent);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
  }
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }

  /* ===== Quote Banner ===== */
  .quote-banner {
    position: relative;
    z-index: 1;
    margin: 2rem 0 2.5rem;
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, rgba(124, 77, 255, 0.06), rgba(124, 77, 255, 0.02));
    border: 1px solid rgba(124, 77, 255, 0.12);
    border-radius: var(--radius-md);
    text-align: center;
    font-family: 'DM Serif Display', 'Noto Serif SC', Georgia, serif;
    font-size: clamp(1rem, 2.5vw, 1.2rem);
    font-style: italic;
    color: var(--text-secondary);
    line-height: 1.7;
    letter-spacing: 0.01em;
  }

  .quote-banner::before {
    content: '"';
    position: absolute;
    top: -0.5rem;
    left: 1rem;
    font-size: 4rem;
    color: var(--accent);
    opacity: 0.25;
    font-family: 'DM Serif Display', serif;
    line-height: 1;
  }

  /* ===== Section Heading ===== */
  .section-heading {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2.5rem 0 1.5rem;
    font-family: 'Noto Sans SC', sans-serif;
    font-weight: 700;
    font-size: clamp(1.1rem, 3vw, 1.35rem);
    color: var(--text-primary);
    letter-spacing: 0.02em;
  }

  .section-heading::before {
    content: '';
    display: block;
    width: 4px;
    height: 1.4em;
    background: var(--accent);
    border-radius: 2px;
  }

  /* ===== Card Grid ===== */
  .nav-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
  }

  .nav-card {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    padding: 1.5rem 1.6rem;
    background: var(--surface-overlay);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    text-decoration: none !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-smooth);
    cursor: pointer;
    overflow: hidden;
    animation: cardReveal 0.5s ease-out both;
  }

  .nav-card:nth-child(1) { animation-delay: 0.05s; }
  .nav-card:nth-child(2) { animation-delay: 0.15s; }
  .nav-card:nth-child(3) { animation-delay: 0.25s; }

  @keyframes cardReveal {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Card accent bar on top */
  .nav-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 3px 3px 0 0;
    transition: height 0.25s ease, opacity 0.25s ease;
  }

  .nav-card.card-ml::before  { background: linear-gradient(90deg, #7c4dff, #b388ff); }
  .nav-card.card-dl::before  { background: linear-gradient(90deg, #448aff, #82b1ff); }
  .nav-card.card-em::before  { background: linear-gradient(90deg, #00bfa5, #64ffda); }

  .nav-card:hover {
    transform: translateY(-6px);
    border-color: rgba(255,255,255,0.12);
    box-shadow: var(--shadow-card-hover);
  }

  .nav-card:hover::before {
    height: 5px;
  }

  /* Card icon */
  .card-icon {
    font-size: 1.8rem;
    line-height: 1;
    margin-bottom: 0.2rem;
  }

  /* Card title */
  .card-title {
    font-family: 'Noto Sans SC', sans-serif;
    font-weight: 700;
    font-size: 1.1em;
    letter-spacing: 0.01em;
    color: var(--text-primary);
  }

  /* Card description */
  .card-desc {
    font-size: 0.88em;
    color: var(--text-secondary);
    line-height: 1.55;
  }

  /* Card arrow */
  .card-arrow {
    align-self: flex-end;
    margin-top: 0.3rem;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    color: var(--text-tertiary);
    transition: all var(--transition-smooth);
  }

  .nav-card:hover .card-arrow {
    background: var(--accent);
    color: #fff;
    transform: translateX(4px);
  }

  /* ===== Responsive ===== */
  @media (max-width: 600px) {
    .nav-grid {
      grid-template-columns: 1fr;
    }
    .terminal-hero {
      border-radius: var(--radius-md);
    }
    .quote-banner {
      padding: 1.2rem 1.5rem;
    }
  }

  /* ===== Dark/Light mode compatibility ===== */
  [data-md-color-scheme="default"] .terminal-hero {
    background: linear-gradient(160deg, #f5f5fa 0%, #eeeef4 40%, #f3f3f8 100%);
    color: #2c2c38;
    border-color: rgba(0,0,0,0.08);
  }
  [data-md-color-scheme="default"] .terminal-header {
    border-bottom-color: rgba(0,0,0,0.06);
  }
  [data-md-color-scheme="default"] .text-output {
    color: #6b6880;
    border-left-color: rgba(0,0,0,0.06);
  }
  [data-md-color-scheme="default"] .nav-card {
    background: #fafafd;
    border-color: rgba(0,0,0,0.06);
  }
  [data-md-color-scheme="default"] .card-desc {
    color: #777590;
  }
  [data-md-color-scheme="default"] .quote-banner {
    background: linear-gradient(135deg, rgba(124, 77, 255, 0.04), rgba(124, 77, 255, 0.01));
    border-color: rgba(124, 77, 255, 0.1);
  }
</style>

<!-- ===== Terminal Hero ===== -->
<div class="hero-wrapper">
  <div class="terminal-hero">
    <div class="terminal-header">
      <div class="dot dot-red"></div>
      <div class="dot dot-yellow"></div>
      <div class="dot dot-green"></div>
      <span class="terminal-title">lmhxts@blog — bash</span>
    </div>
    <div class="terminal-body">
      <div class="line">
        <span class="prompt">lmhxts@blog:~$</span>
        <span class="command">whoami</span>
      </div>
      <span class="text-output">LI Muhang — physics & ML learner, note-taker, curious mind.</span>

      <div class="line">
        <span class="prompt">lmhxts@blog:~$</span>
        <span class="command">./explore.sh --help</span>
      </div>
      <span class="text-output">Usage: explore [--ml] [--dl] [--em] [--all]</span>

      <div class="line">
        <span class="prompt">lmhxts@blog:~$</span>
        <span class="command">./explore.sh --all<span class="cursor-blink"></span></span>
      </div>
      <span class="sys-output">✔ Loading Machine Learning Modules... [OK]</span>
      <span class="sys-output">✔ Initializing Deep Learning Networks... [OK]</span>
      <span class="sys-output">✔ Spooling Electrodynamics Notes... [OK]</span>
      <span class="sys-output">✔ System Ready. Enjoy exploring!</span>
    </div>
  </div>
</div>

<!-- ===== Quote ===== -->
<div class="quote-banner">
  Sich selber übervoll, sich selber bethauen,<br>
  sich selber Regenguss sein einer verschmachtenden Wildniss.
</div>

<!-- ===== Quick Navigation ===== -->
<div class="section-heading">🔭 探索笔记</div>

<div class="nav-grid">

  <a href="ml/Bayes/" class="nav-card card-ml">
    <span class="card-icon">🧠</span>
    <span class="card-title">机器学习</span>
    <span class="card-desc">
      贝叶斯学习、决策树、XGBoost 数学推导、逻辑回归、K-近邻、CART 等经典算法笔记。
    </span>
    <span class="card-arrow">→</span>
  </a>

  <a href="dl/BPNN/" class="nav-card card-dl">
    <span class="card-icon">🔮</span>
    <span class="card-title">深度学习</span>
    <span class="card-desc">
      BP 神经网络、优化与泛化理论，从反向传播到现代深度学习的核心概念。
    </span>
    <span class="card-arrow">→</span>
  </a>

  <a href="electrodynamics/talkemd_2/" class="nav-card card-em">
    <span class="card-icon">⚡</span>
    <span class="card-title">电动力学</span>
    <span class="card-desc">
      麦克斯韦方程组、电磁场边值关系、坡印亭矢量——从场的观点理解电磁现象。
    </span>
    <span class="card-arrow">→</span>
  </a>

</div>

<br>

## 👋 欢迎

这里是我记录课内外学习过程的地方，包含机器学习、深度学习的算法推导、数学笔记和一些其他思考，目前仍在不断更新和完善。

> *"The noblest pleasure is the joy of understanding."* — Leonardo da Vinci
