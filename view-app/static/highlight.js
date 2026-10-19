/* Local Prism renderer. Tokens are escaped; source text is never treated as HTML. */
const CodeView = (() => {
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const aliases = {py:'python',sh:'bash',shell:'bash',zsh:'bash',js:'javascript',mjs:'javascript',cjs:'javascript',yml:'yaml',html:'markup',xml:'markup',patch:'diff',jsonl:'json'};
  function language(text, hint='') {
    const name=hint.toLowerCase();
    const extension=name.split('.').pop();
    const explicit=aliases[extension]||extension;
    if(Prism.languages[explicit])return explicit;
    if(/^(command|cmd|shell_command|bash_command)$/.test(name))return 'bash';
    if(/^#!.*\bpython/.test(text)||/^(?:from [\w.]+ import |import \w|(?:async )?def \w+\(|class \w+[:(])/m.test(text))return 'python';
    if(/^#!.*\b(?:ba|z)?sh\b/.test(text))return 'bash';
    if(/^(?:diff --git |\*\*\* Begin Patch|@@ )/m.test(text))return 'diff';
    if(/^[\s]*[\[{]/.test(text)){try{JSON.parse(text);return 'json';}catch(_){}}
    return 'text';
  }
  function tokens(text,lang) {
    const grammar=Prism.languages[lang];
    if(!grammar)return [text];
    // Python heredocs are common in coding-agent shell calls. Keep the shell
    // wrapper in Bash, and tokenize the embedded program as Python.
    if(lang==='bash'){
      const heredoc=/\bpython(?:3(?:\.\d+)?)?\b[^\n]*?<<-?\s*['"]?([A-Za-z_]\w*)['"]?[^\n]*\n([\s\S]*?)\n\1(?=\r?\n|$)/g;
      let match,start=0,result=[];
      while((match=heredoc.exec(text))){
        const bodyStart=match.index+match[0].indexOf('\n')+1;
        result.push(...Prism.tokenize(text.slice(start,bodyStart),grammar));
        result.push(...Prism.tokenize(match[2],Prism.languages.python));
        start=bodyStart+match[2].length;
      }
      if(result.length){result.push(...Prism.tokenize(text.slice(start),grammar));return result;}
    }
    return Prism.tokenize(text,grammar);
  }
  function lines(text,lang) {
    const result=[''];
    function walk(token,classes=[]) {
      if(Array.isArray(token)){token.forEach(t=>walk(t,classes));return;}
      if(typeof token!=='string'){
        const names=[token.type,...(Array.isArray(token.alias)?token.alias:token.alias?[token.alias]:[])];
        walk(token.content,[...classes,...names]);return;
      }
      token.split('\n').forEach((part,index)=>{
        if(index)result.push('');
        const value=escape(part);
        result[result.length-1]+=classes.length?`<span class="token ${escape(classes.join(' '))}">${value}</span>`:value;
      });
    }
    walk(tokens(text,lang));
    return result;
  }
  function render(value,hint='') {
    const text=String(value??'');
    const lang=language(text,hint);
    return `<div class="code-editor" data-language="${lang}"><div class="code-caption">${escape(lang==='markup'?'HTML / XML':lang==='text'?'纯文本':lang)}<button type="button" class="code-fullscreen" title="全屏查看代码">全屏 ⛶</button></div><pre class="syntax-code"><code>${lines(text,lang).map(line=>`<span class="code-line">${line}</span>`).join('\n')}</code></pre></div>`;
  }
  const fullscreen=document.createElement('dialog');
  fullscreen.className='code-fullscreen-dialog';
  fullscreen.setAttribute('aria-label','全屏代码查看');
  document.body.append(fullscreen);
  let fullscreenOpener=null;
  fullscreen.addEventListener('close',()=>{
    document.body.classList.remove('code-fullscreen-open');
    fullscreen.replaceChildren();
    if(fullscreenOpener?.isConnected)fullscreenOpener.focus({preventScroll:true});
    fullscreenOpener=null;
  });
  document.addEventListener('click',async event=>{
    const open=event.target.closest('.code-fullscreen');
    if(open){
      const editor=open.closest('.code-editor');
      const title=editor.closest('#preview')?.querySelector('.file-breadcrumb')?.title
        ||editor.closest('.event')?.querySelector('.snippet')?.textContent||'代码查看';
      fullscreenOpener=open;
      fullscreen.innerHTML=`<div class="fullscreen-toolbar"><strong>${escape(title)}</strong><span>Esc 退出</span><button type="button" class="fullscreen-copy">复制代码</button><button type="button" class="close-fullscreen" autofocus>退出全屏</button></div>`;
      const clone=editor.cloneNode(true);
      clone.querySelector('.code-fullscreen').remove();
      fullscreen.append(clone);
      fullscreen.showModal();
      document.body.classList.add('code-fullscreen-open');
      clone.querySelector('pre').scrollTop=editor.querySelector('pre').scrollTop;
    }
    if(event.target.closest('.close-fullscreen'))fullscreen.close();
    const copy=event.target.closest('.fullscreen-copy');
    if(copy){
      try{
        await navigator.clipboard.writeText(fullscreen.querySelector('code').textContent);
        copy.textContent='已复制';
      }catch(_){copy.textContent='复制失败，请选择文本复制';}
    }
  });
  const wrapButton=document.getElementById('global-wrap');
  function setWrap(enabled) {
    document.documentElement.dataset.wrap=String(enabled);
    if(wrapButton){
      wrapButton.setAttribute('aria-pressed',String(enabled));
      wrapButton.textContent=`自动换行：${enabled?'开':'关'}`;
    }
  }
  let savedWrap=true;
  try {savedWrap=localStorage.getItem('posttrain-wrap')!=='false';} catch (_) {}
  setWrap(savedWrap);
  if(wrapButton)wrapButton.addEventListener('click',()=>{
    const enabled=document.documentElement.dataset.wrap!=='true';
    setWrap(enabled);
    try {localStorage.setItem('posttrain-wrap',String(enabled));} catch (_) {}
  });
  return {render,language};
})();
