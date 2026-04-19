---
hide:
  - navigation
  - toc
---

<style>
  .terminal-hero {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 20px;
    font-family: 'Consolas', 'Fira Code', monospace;
    color: #e0e0e0;
    margin-top: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    /* 新增：优化长文本的自动换行，防止撑破容器 */
    word-break: break-word;
    white-space: normal; 
  }
  .terminal-header {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
  }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .dot-red { background: #ff5f56; }
  .dot-yellow { background: #ffbd2e; }
  .dot-green { background: #27c93f; }
  
  .terminal-body { font-size: 1.1em; line-height: 1.6; }
  .prompt { color: #5cc5ef; font-weight: bold; margin-right: 8px; }
  .command { color: #f8f8f2; }
  
  
  .text-output { 
    color: #a0a0a0; 
    font-style: italic; 
    margin-top: 4px;
    margin-bottom: 16px; 
    display: block; 
  } 
  .sys-output { color: #a6e22e; margin-top: 4px;}
</style>

<div class="terminal-hero">
  <div class="terminal-header">
    <div class="dot dot-red"></div>
    <div class="dot dot-yellow"></div>
    <div class="dot dot-green"></div>
  </div>
  <div class="terminal-body">
    <div><span class="prompt">lmhxts@blog:~$</span><span class="command">echo "Sich selber übervoll, sich selber bethauen, sich selber Regenguss sein einer verschmachtenden Wildniss."</span></div>
    
    <div class="text-output">Sich selber übervoll, sich selber bethauen, sich selber Regenguss sein einer verschmachtenden Wildniss.</div>
    
    <div style="margin-top: 8px;"><span class="prompt">lmhxts@blog:~$</span><span class="command">./start_learning.sh</span></div>
    <div class="sys-output">> Loading Machine Learning Modules... [OK]</div>
    <div class="sys-output">> Initializing Neural Networks... [OK]</div>
    <div class="sys-output">> System Ready. Enjoy reading!</div>
  </div>
</div>

<br><br>

## 👋 欢迎来到我的blog

这里是我记录课内外学习过程的地方，包含机器学习、深度学习的算法、数学推导和一些其他想法，目前仍在不断更新和完善。

你可以通过顶部的导航栏，或者点击下方的快捷链接开始阅读：

* 🧭 [前往阅读：机器学习笔记](ml/Bayes.md)
* 🧠 [前往阅读：深度学习笔记](dl/BPNN.md)
* 🙋‍♂️ [关于我](about.md)