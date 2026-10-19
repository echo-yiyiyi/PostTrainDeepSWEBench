const $ = id => document.getElementById(id);
const state = {runs:[], run:null, summary:null, offset:0, generation:0, traceRequest:0, fileRequest:0, treeRequest:0, fileScope:'workspace'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty = value => typeof value === 'string' ? value : JSON.stringify(value,null,2);
const size = bytes => bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes/1024).toFixed(1)} KB` : `${(bytes/1048576).toFixed(1)} MB`;
function error(e) { $('error').textContent=e.message; $('error').hidden=false; }
function safe(fn) {return (...args)=>Promise.resolve().then(()=>fn(...args)).catch(error);}
async function api(path, params={}) {const response=await fetch(`/api/${path}?`+new URLSearchParams({run:state.run||'.',...params}));const data=await response.json();if(!response.ok)throw new Error(data.error);return data;}
function code(text,hint='') {return CodeView.render(text,hint);}
function payload(label,text,depth=0,hint='') {
  let value=text;
  if(typeof value==='string') {try {value=JSON.parse(value);} catch (_) {}}
  if(depth<5 && value && typeof value==='object' && !Array.isArray(value)) {
    return `<div class="field-label">${esc(label)}</div><div class="structured">`+Object.entries(value).map(([key,item])=>payload(key,item,depth+1,key==='content'?(value.file_path||value.path||key):key)).join('')+'</div>';
  }
  if(Array.isArray(value) && value.length && value.every(item=>item?.type==='text' && typeof item.text==='string')) value=value.map(item=>item.text).join('\n');
  return `<div class="field-label">${esc(label)}<button class="copy">复制</button></div>${code(pretty(value),hint||label)}`;
}
function tab(name) {document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));['overview','trace','evaluation','files'].forEach(id=>$(id).hidden=id!==name);}
function renderRuns() {const query=$('run-search').value.toLowerCase();const rows=state.runs.filter(r=>r.id.toLowerCase().includes(query));$('runs').innerHTML=rows.map(r=>`<button data-run="${esc(r.id)}" class="${state.run===r.id?'active':''}"><small>${esc(r.group)}</small>${esc(r.name)}</button>`).join('')||'<p class="muted">没有匹配的实验。可用 --results-dir 指定结果目录。</p>';$('runs').querySelectorAll('button').forEach(b=>b.onclick=safe(()=>selectRun(b.dataset.run)));}
async function loadRuns() {$('error').hidden=true;const data=await api('runs');state.runs=data.runs;$('root').textContent=data.root;renderRuns();if(state.runs.length){const selected=state.runs.find(r=>r.id===state.run)||state.runs[0];await selectRun(selected.id);}}
async function selectRun(id) {evaluationRequest++;evaluationLoaded.clear();$('eval-calls').innerHTML='';$('eval-source').innerHTML='';state.run=id;const generation=++state.generation;state.offset=0;state.fileRequest++;state.traceRequest++;state.summary=null;$('error').hidden=true;renderRuns();$('title').textContent=state.runs.find(r=>r.id===id)?.name||id;$('subtitle').textContent=id;['overview','events','tree'].forEach(x=>$(x).innerHTML='<p class="muted">加载中…</p>');$('preview').innerHTML='<div class="empty">选择文件查看内容</div>';const data=await api('summary');if(generation!==state.generation)return;state.summary=data;renderOverview();$('trace-source').innerHTML=data.traces.map(p=>`<option value="${esc(p)}">${esc(p)}</option>`).join('');$('eval-source').innerHTML=data.evaluations.map(e=>`<option value="${esc(e.path)}">${esc(e.phase)} · ${esc(e.path)}</option>`).join('');await Promise.all([loadTrace(),loadTree(),loadEvaluation()]);}
function renderOverview() {const docs=state.summary.documents;const metrics=docs['metrics_averaged_exact_match.json']||docs['metrics_averaged.json']||docs['metrics.json'];let cards='';const status=docs['status.json']?.json?.status;cards+=`<div class="card"><small>运行状态</small><b>${esc(status|| (metrics?'已有指标':'未记录'))}</b></div>`;cards+=`<div class="card"><small>Agent 执行时间</small><b>${esc(docs['time_taken.txt']?.text.trim()||'—')}</b></div>`;for(const [key,value] of Object.entries(metrics?.json||{})){if(typeof value==='number')cards+=`<div class="card"><small>${esc(key)}</small><b>${Number.isInteger(value)?value:Number(value.toFixed(5))}</b></div>`;}
$('overview').innerHTML=`<div class="cards">${cards}</div><div class="note">工作目录：${esc(state.summary.workspaces.join(' · ')||'尚未保存')}。这里展示的是已保存的产物；标准流程中的 task 为清理后的快照。指标按源文件显示，未在此重新计算 judge 回退得分。</div><h2>实验记录与评估</h2>`+Object.entries(docs).map(([name,data])=>`<details class="doc"><summary>${esc(name)}${data.error?' · 读取提示':''}</summary><div class="content">${data.error?code(data.error):''}${payload('内容',data.text||'')}${data.more?'<p class="muted">概览仅展示前 256 KB；可在文件浏览中继续阅读。</p>':''}</div></details>`).join('');}
async function loadTrace() {const request=++state.traceRequest;const path=$('trace-source').value;if(!path){$('events').innerHTML='<div class="empty">这次实验还没有保存轨迹。</div>';$('trace-count').textContent='0 个事件';$('prev').disabled=$('next').disabled=true;return;}$('events').innerHTML='<p class="muted">解析轨迹中…</p>';const data=await api('trace',{path,offset:state.offset,search:$('trace-search').value,kind:$('kind').value});if(request!==state.traceRequest)return;$('trace-count').textContent=`${data.total?state.offset+1:0}–${Math.min(state.offset+100,data.total)} / ${data.total} 个事件 · 全部 ${data.all_total}`;$('prev').disabled=state.offset===0;$('next').disabled=state.offset+100>=data.total;
$('events').innerHTML=data.events.map((e,i)=>`<details class="event ${esc(e.kind)}"><summary><span class="type">${esc(e.kind)}</span><span class="snippet">${esc(pretty(e.title))} ${esc(e.body.replace(/\s+/g,' ').slice(0,160))}</span><span class="badge">${esc(e.status)}</span><time>${esc(e.timestamp)} · L${e.line}</time></summary><div class="content" data-event="${i}"></div></details>`).join('')||'<div class="empty">没有匹配的事件</div>';
$('events').querySelectorAll('details.event').forEach((el,i)=>el.ontoggle=()=>{if(!el.open||el.dataset.loaded)return;el.dataset.loaded='1';const e=data.events[i];el.querySelector('.content').innerHTML=(e.body?payload(e.kind==='tool'?'输入 / 参数':'内容',e.body,0,e.kind==='tool'&&(/command_execution|exec_command|shell|bash/i.test(pretty(e.raw))||/^\/?(?:usr\/)?bin\/(?:ba)?sh/.test(String(e.title)))?'bash':''):'')+(e.output?payload('工具输出',e.output):'')+`<details class="raw"><summary>原始事件 JSON (${e.raw.length})</summary>${code(pretty(e.raw),'json')}</details>`;});}
let evaluationRequest=0;
const evaluationLoaded=new Map();
function parseToolArguments(value) {
  if(typeof value!=='string')return {value, warning:value===null||typeof value!=='object'?'工具参数不是对象':''};
  try {
    const parsed=JSON.parse(value);
    return {value:parsed,warning:parsed===null||typeof parsed!=='object'?'工具参数不是对象':''};
  } catch (_) {return {value,warning:'参数 JSON 格式错误，以下保留原文'};}
}
function evaluationEvents(messages) {
  const events=[], tools=new Map();
  const add=(kind,title,body,extra={})=>{const e={kind,title,body,...extra};events.push(e);return e;};
  for(const message of messages){
    if(!message||typeof message!=='object'){add('log','未识别的消息',message);continue;}
    const role=message.role||'system';
    if(role==='tool'){
      const id=message.tool_call_id;
      let result=message.content;
      if(typeof result==='string'){try{result=JSON.parse(result);}catch(_){}}
      const tool=id?tools.get(id):null;
      const event=tool||add('tool',message.name||'工具返回（无匹配调用）','');
      event.output=result;
      const exit=result?.returncode??result?.exit_code??message.extra?.returncode;
      event.status=exit!=null?`exit ${exit}`:message.is_error?'error':'已返回';
      if(tool)tools.delete(id);
      continue;
    }
    if(message.reasoning_content)add('reasoning','思考',message.reasoning_content);
    if(message.content!=null&&message.content!=='')add(role,role,message.content);
    let calls=message.tool_calls;
    if(calls==null&&message.function_call)calls=[message.function_call];
    if(calls==null)continue;
    if(!Array.isArray(calls)){add('tool','未识别的工具调用',calls,{warning:'tool_calls 不是数组，以下保留原值'});continue;}
    for(const call of calls){
      if(!call||typeof call!=='object'){add('tool','未识别的工具调用',call,{warning:'工具调用结构无效，以下保留原值'});continue;}
      const fn=call.function&&typeof call.function==='object'?call.function:call;
      const args=parseToolArguments(fn.arguments??fn.input??'');
      const e=add('tool',fn.name||(typeof call.function==='string'?call.function:'tool'),args.value,{warning:args.warning,status:'调用',id:call.id});
      if(call.id)tools.set(call.id,e);
    }
  }
  return events;
}
function evalEventCard(event) {
  const kind=['assistant','user','system','tool','reasoning','log'].includes(event.kind)?event.kind:'system';
  const text=typeof event.body==='string'?event.body:pretty(event.body);
  const preview=(text||'').replace(/\s+/g,' ').slice(0,180);
  let body='';
  if(event.warning)body+=`<p class="eval-warning">${esc(event.warning)}</p>`;
  if(event.body!=null&&event.body!=='')body+=event.warning?code(typeof event.body==='string'?event.body:pretty(event.body),'text'):payload(kind==='tool'?'输入 / 参数':'内容',event.body,0,kind==='tool'&&/bash|shell|exec/.test(event.title)?'bash':'');
  if(event.output!==undefined)body+=payload('工具输出',event.output);
  return `<details class="event ${kind}"><summary><span class="type">${esc(kind)}</span><span class="snippet">${esc(event.title)} ${esc(preview)}</span>${event.status?`<span class="badge">${esc(event.status)}</span>`:''}${event.warning?'<span class="eval-warning-badge">格式异常</span>':''}</summary><div class="content">${body}</div></details>`;
}
function evalMessageList(messages) {return evaluationEvents(messages).map(evalEventCard).join('');}
function evaluationOutputs(output) {
  if(output?.role)return [output];
  if(Array.isArray(output?.choices))return output.choices.map(choice=>choice.message||{role:'assistant',content:choice.text??choice});
  return [{role:'assistant',content:output?.completion??output}];
}
function renderEvaluationCall(content,call) {
  if($('eval-format').value==='raw'){
    content.innerHTML='<p class="muted">保存的消息字段与返回值（JSON 展示）；参数字符串不修复、不改写。输入上下文来源见 basis。</p>'+code(pretty(call),'json');
    return;
  }
  const inputs=Array.isArray(call.input)?call.input:[];
  content.innerHTML=`<p class="muted">${esc(call.basis)}</p><details class="doc eval-input"><summary>本轮输入上下文 · ${inputs.length} 条消息</summary><div class="content"></div></details><h2>模型输出与工具调用</h2>`+
    evalMessageList([...evaluationOutputs(call.output),...(call.following||[])]);
  // Long input histories are formatted only when expanded.
  const input=content.querySelector('.eval-input');
  input.ontoggle=()=>{if(input.open&&!input.dataset.loaded){input.dataset.loaded='true';input.querySelector('.content').innerHTML=inputs.length?evalMessageList(inputs):payload('输入',call.input);}};
}
async function loadEvaluation() {
  const request=++evaluationRequest,generation=state.generation,path=$('eval-source').value;
  evaluationLoaded.clear();
  $('eval-calls').innerHTML='';
  if(!path){$('eval-description').textContent='本次实验未找到已保存的评测对话。';return;}
  $('eval-description').textContent='读取评测记录…';
  const data=await api('evaluation',{path});
  if(request!==evaluationRequest||generation!==state.generation)return;
  $('eval-description').textContent=`${data.calls.length} 次模型回复 · ${data.description}。工具调用与返回按 ID 合并，可切换原始 JSON 检查格式异常。`;
  $('eval-calls').innerHTML=data.calls.map(c=>`<details class="doc eval-call"><summary>第 ${c.index+1} 轮 ${c.sample?'· '+esc(c.sample):''} · ${esc(c.preview.replace(/\s+/g,' '))}</summary><div class="content"></div></details>`).join('')||'<div class="empty">此记录没有模型调用；可能只包含评分或评测未完成。</div>';
  $('eval-calls').querySelectorAll('.eval-call').forEach((el,index)=>el.ontoggle=safe(async()=>{
    if(!el.open||el.dataset.loaded)return;
    el.dataset.loaded='loading';
    const content=el.querySelector('.content');content.textContent='加载本轮对话…';
    try{
      const call=await api('evaluation',{path,call:index});
      if(request!==evaluationRequest||generation!==state.generation)return;
      evaluationLoaded.set(content,call);
      renderEvaluationCall(content,call);
      el.dataset.loaded='true';
    }catch(e){delete el.dataset.loaded;content.textContent=e.message;}
  }));
}
$('eval-source').onchange=safe(loadEvaluation);
$('eval-format').onchange=()=>{for(const [content,call] of evaluationLoaded)if(content.isConnected)renderEvaluationCall(content,call);};
function fileIcon(name,directory=false) {
  if(directory)return '<svg class="fs-icon folder-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v10H3z"/></svg>';
  const ext=name.split('.').pop().toLowerCase();
  const types={py:['PY','python'],sh:['$_','shell'],bash:['$_','shell'],json:['{}','config'],jsonl:['{}','config'],yaml:['Y','config'],yml:['Y','config'],toml:['T','config'],js:['JS','javascript'],ts:['TS','javascript'],md:['M↓','text'],txt:['≡','text'],log:['≡','text']};
  const [label,type]=types[ext]||['·','other'];
  return `<span class="fs-icon file-icon ${type}" aria-hidden="true">${label}</span>`;
}
function pathCrumbs(path) {
  return `<span class="file-breadcrumb" title="${esc(path)}">`+path.split('/').filter(Boolean).map(part=>`<span>${esc(part)}</span>`).join('<i aria-hidden="true">/</i>')+'</span>';
}
async function fillFolder(path,container,generation,scope,treeRequest) {
  const data=await api('files',{path,scope});
  if(generation!==state.generation||treeRequest!==state.treeRequest)return;
  container.innerHTML='';
  if(!data.entries.length){container.innerHTML='<div class="tree-empty">空文件夹</div>';return;}
  for(const entry of data.entries){
    if(entry.directory){
      const details=document.createElement('details');
      details.className='tree-folder';
      details.innerHTML=`<summary title="${esc(entry.path)}"><span class="tree-arrow" aria-hidden="true">›</span>${fileIcon(entry.name,true)}<span class="entry-name">${esc(entry.name)}</span>${entry.symlink?'<span class="link-mark" title="符号链接">↗</span>':''}</summary><div class="tree-children"></div>`;
      details.ontoggle=safe(async()=>{if(details.open&&!details.dataset.loaded){await fillFolder(entry.path,details.querySelector('div'),generation,scope,treeRequest);details.dataset.loaded='1';}});
      container.append(details);
    }else{
      const b=document.createElement('button');
      b.className='tree-file';
      b.innerHTML=`<span class="tree-arrow spacer"></span>${fileIcon(entry.name)}<span class="entry-name">${esc(entry.name)}</span>${entry.symlink?'<span class="link-mark">↗</span>':''}<span class="file-size">${entry.blocked?'⊘':size(entry.size)}</span>`;
      b.title=entry.error||size(entry.size);b.disabled=!!entry.blocked;
      b.onclick=safe(()=>{container.closest('#tree').querySelectorAll('button').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');return openFile(entry.path,0,scope);});
      container.append(b);
    }
  }
}
async function loadTree(scope='workspace') {
  state.fileScope=scope;
  const request=++state.treeRequest;
  state.fileRequest++;
  $('preview').innerHTML='<div class="empty">点击左侧文件查看代码或文本内容。</div>';
  $('file-task').classList.toggle('active',scope==='workspace');
  $('file-root').classList.toggle('active',scope!=='workspace');
  $('workspace-path').textContent=scope==='workspace'?'/home/ben/task/':'实验记录文件';
  $('workspace-note').textContent=scope==='workspace'?'Agent 工作目录中最终保留下来的文件；不包含已删除文件和目录外的依赖、缓存。':'运行日志、轨迹、评估指标等实验记录。';
  if(!state.summary)return;
  if(scope==='workspace'&&!state.summary.workspace){$('tree').innerHTML='<p class="muted">本次实验未保存工作目录。</p>';return;}
  $('tree').innerHTML=`<div class="tree-heading"><span>文件资源管理器</span><button id="collapse-folders" title="收起所有文件夹">全部收起</button></div><div class="tree-root">${fileIcon('task',true)}<b>${scope==='workspace'?'task':'实验记录'}</b><span class="muted">/</span></div><div id="tree-entries"><p class="muted">加载目录…</p></div>`;
  $('collapse-folders').onclick=()=>$('tree').querySelectorAll('details').forEach(folder=>folder.open=false);
  await fillFolder('.',$('tree-entries'),state.generation,scope,request);
}
async function openFile(path,offset=0,scope=state.fileScope) {
  const request=++state.fileRequest;
  const generation=state.generation;
  const data=await api('file',{path,offset,scope});
  if(request!==state.fileRequest||generation!==state.generation)return;
  const displayPath=scope==='workspace'?`/home/ben/task/${path}`:path;
  $('preview').innerHTML=`<h3>${pathCrumbs(displayPath)}</h3><p class="muted">${size(data.size)} · 字节 ${offset}–${data.next_offset}</p>`+(data.binary?'<div class="empty">二进制文件 · 不进行文本渲染</div>':`<button id="copy-file">复制当前段</button>${code(data.text,path)}`)+`<div class="toolbar"><button id="file-prev" ${offset===0?'disabled':''}>上一段</button><button id="file-next" ${!data.more?'disabled':''}>下一段</button>${/\.(jsonl|json|txt)$/.test(path)?'<button id="as-trace">作为轨迹查看</button>':''}</div>`;
  $('file-prev').onclick=safe(()=>openFile(path,Math.max(0,offset-262144),scope));
  $('file-next').onclick=safe(()=>openFile(path,data.next_offset,scope));
  if($('copy-file'))$('copy-file').onclick=safe(()=>navigator.clipboard.writeText(data.text));
  if($('as-trace'))$('as-trace').onclick=safe(async()=>{
    const source=scope==='workspace'?`${state.summary.workspace.source}/${path}`:path;
    if(!Array.from($('trace-source').options).some(o=>o.value===source))$('trace-source').add(new Option(displayPath,source));
    $('trace-source').value=source;state.offset=0;tab('trace');await loadTrace();
  });
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>tab(b.dataset.tab));$('run-search').oninput=renderRuns;$('refresh').onclick=safe(loadRuns);$('trace-source').onchange=$('kind').onchange=safe(()=>{state.offset=0;return loadTrace();});let timer;$('trace-search').oninput=()=>{clearTimeout(timer);timer=setTimeout(safe(()=>{state.offset=0;return loadTrace();}),300);};$('prev').onclick=safe(()=>{state.offset=Math.max(0,state.offset-100);return loadTrace();});$('next').onclick=safe(()=>{state.offset+=100;return loadTrace();});$('expand').onclick=()=>document.querySelectorAll('details.event').forEach(e=>e.open=true);$('collapse').onclick=()=>document.querySelectorAll('details.event').forEach(e=>e.open=false);$('file-root').onclick=safe(()=>loadTree('artifacts'));$('file-task').onclick=safe(()=>loadTree('workspace'));document.addEventListener('click',safe(async event=>{if(event.target.matches('.copy')){await navigator.clipboard.writeText(event.target.parentElement.nextElementSibling.querySelector('code').textContent);event.target.textContent='已复制';}}));safe(loadRuns)();
