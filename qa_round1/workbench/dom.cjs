/* Production HTML and JS served by the isolated HTTP server. Browser-only APIs are simulated. */
const fs = require('node:fs'),path = require('node:path'),vm = require('node:vm');
const assert=require('node:assert/strict');
const {VH_QA_BASE:base,VH_QA_DIR:tmp,VH_QA_VIDEO:video,VH_QA_MONO:mono}=process.env;
let parseHTML;
for(const name of [process.env.LINKEDOM_MODULE,'linkedom','/opt/homebrew/lib/node_modules/openclaw/node_modules/linkedom'].filter(Boolean)){
  try{({parseHTML}=require(name));break;}catch{}
}
if(!parseHTML) throw Error('Set LINKEDOM_MODULE to a local linkedom installation.');
const report={execution:'DOM event/handler simulation + real HTTP/FFmpeg; layout and real playback unverified here',checks:[]};
const record=(id,pass,details={})=>report.checks.push({id,status:pass?'pass':'fail',...details});
const wait=async(fn,explain='state')=>{const stop=Date.now()+50000;while(!fn()&&Date.now()<stop) await new Promise(r=>setTimeout(r,30));assert(fn(),explain);};
async function setup(search){
  const {document,window}=parseHTML(await (await fetch(base)).text());
  const $=id=>document.getElementById(id);
  const inp=Object.getPrototypeOf($('start')),sel=Object.getPrototypeOf($('mode'));
  Object.defineProperty(inp,'valueAsNumber',{configurable:true,get(){return this.value===''?NaN:Number(this.value);}});
  Object.defineProperty(sel,'value',{configurable:true,get(){return this.querySelector('[selected]')?.value??this.options[0]?.value??'';},set(value){Array.from(this.options).forEach(x=>x.toggleAttribute('selected',x.value===value));}});
  const canvas=$('wave');Object.assign(canvas,{getContext:()=>new Proxy({},{get:()=>()=>{},set:()=>true})});
  Object.defineProperty(canvas,'clientWidth',{value:720});Object.defineProperty(canvas,'clientHeight',{value:96});
  const audio=$('player');Object.assign(audio,{paused:true,readyState:0,ended:false,currentTime:0,load(){this.readyState=1;this.dispatchEvent(new window.Event('loadedmetadata'));},pause(){this.paused=true;this.dispatchEvent(new window.Event('pause'));},play(){this.paused=false;this.dispatchEvent(new window.Event('play'));return Promise.resolve();}});
  const ctx=vm.createContext({document,window,location:new URL(search||'',base),URL,URLSearchParams,ResizeObserver:class{observe(){}},Option:class{constructor(t,v){let o=document.createElement('option');o.textContent=t;o.value=v;return o;}},encodeURIComponent,fetch:(u,o)=>fetch(new URL(u,base),o)});
  vm.runInContext(await (await fetch(new URL('/app.js',base))).text(),ctx);
  const fire=(id,event)=>$(id).dispatchEvent(new window.Event(event));
  return {$,audio,window,fire};
}
async function main(){
  if(process.argv.includes('--no-phone')){
    const {$}=await setup('');await wait(()=>$('mode').options.length>0);
    const caps=await (await fetch(new URL('/api/capabilities',base))).json();
    record('private-fixture-button-hidden-when-unconfigured',caps.iphone_demo===false&&$('iphone-demo').hidden);
    const {document}=parseHTML(await (await fetch(base)).text());
    const css=await (await fetch(new URL('/style.css',base))).text();
    const controls=Array.from(document.querySelectorAll('input, select, button'));
    const unlabelled=controls.filter(e=>!e.hasAttribute('aria-label')&&!e.closest('label')&&!e.textContent.trim()&&!document.querySelector('label[for="'+e.id+'"]'));
    record('static-responsive-and-control-markup',unlabelled.length===0&&css.includes('@media(max-width:700px)')&&css.includes('@media(max-width:380px)')&&!!document.querySelector('meta[name="viewport"]'),{checkedControls:controls.length,unlabelled:unlabelled.map(e=>e.id),scope:'markup and CSS rules only; real layout and taps not measured'});
    fs.writeFileSync(path.join(tmp,'no-phone.json'),JSON.stringify(report,null,2));return;
  }
  for(const [source,filename,mode] of [['studio','studio.wav','example'],['duet','duet.wav','speakers'],['iphone',path.basename(video),'rnnoise']]){
    const {$}=await setup('?demo='+source);await wait(()=>$('filename').textContent===filename,source+' deep link');
    record('sample-link-'+source,$('mode').value===mode,{selected:$('mode').value,syntheticPhoneFixture:source==='iphone'});
  }
  const {$,audio,fire}=await setup('');await wait(()=>$('mode').options.length>0);
  const upload=async(filepath)=>{
    const file=new File([fs.readFileSync(filepath)],path.basename(filepath),{type:'application/octet-stream'});
    Object.defineProperty($('file'),'files',{value:[file],configurable:true});
    $('file').onchange();await wait(()=>!$('file').disabled,'import busy state');
  };
  await upload(video);record('unicode-mov-upload',$('filename').textContent===path.basename(video)&&!$('separation-settings').disabled);
  await upload(mono);record('mono-error-and-prior-clip-usable',$('notice').classList.contains('error')&&$('filename').textContent===path.basename(video)&&!$('separation-settings').disabled);
  $('mode').value='example';$('mode').onchange();
  $('start').value='2';$('end').value='1';await $('separate').onclick();
  record('invalid-span-before-first-separation',$('notice').classList.contains('error')&&$('stems').hidden&&!$('separation-settings').disabled);
  $('start').value='.05';$('end').value='2.05';$('use-background').checked=true;$('use-background').onchange();
  $('bg-start').value='1.5';$('bg-end').value='3.5';await $('separate').onclick();
  record('overlapping-spans-error',$('notice').classList.contains('error')&&$('stems').hidden&&!$('separation-settings').disabled);
  $('bg-start').value='2.35';$('bg-end').value='4.30';await $('separate').onclick();
  assert(!$('stems').hidden,$('notice').textContent);
  const stemURL=$('dl-target').href;record('valid-separation-after-error',!!stemURL&&!$('mix-settings').disabled);
  $('solo-target').onclick();const t=audio.src;$('solo-residual').onclick();const r=audio.src;
  record('solo-target-residual-URL-switch',t.endsWith('-target.wav')&&r.endsWith('-residual.wav'));
  $('seek').value='3.25';$('seek').oninput();$('listen-original').onclick();const origURL=audio.src;
  const posA=audio.currentTime;
  $('pan').value='0.7';$('gain').value='-6';$('width').value='1.45';
  ['pan','gain','width'].forEach(x=>fire(x,'input'));await $('render').onclick();
  const firstMix=$('dl-mix').href;
  $('seek').value='3.25';$('seek').oninput();$('listen-original').onclick();$('listen-mix').onclick();
  record('AB-source-and-position-request',audio.src.endsWith('-mix.wav')&&Math.abs(posA-3.25)<.01&&Math.abs(audio.currentTime-3.25)<.01,{seek:audio.currentTime});
  $('gain').value='-8';fire('gain','input');
  record('stale-render-explicit',!$('stale-mix').hidden&&$('dl-mix').href===firstMix);
  await $('render').onclick();record('current-render-file',$('dl-mix').href!==firstMix&&$('stale-mix').hidden);
  await $('export-video').onclick();assert(!$('dl-video').hidden,$('notice').textContent);
  let downloads={original:origURL,target:$('dl-target').href,residual:$('dl-residual').href,mix:$('dl-mix').href,video:$('dl-video').href};
  for(const [id,url] of Object.entries(downloads)){
    const response=await fetch(new URL(url,base));
    const bytes=Buffer.from(await response.arrayBuffer());assert.equal(response.status,200);assert(bytes.length>2048);
    fs.writeFileSync(path.join(tmp,id+(id==='video'?'.mov':'.wav')),bytes);
  }
  record('complete-downloads',true,{files:5,videoDownloadName:$('dl-video').download});
  const was = $('notice').textContent;
  $('end').value='1.35';fire('end','input');
  function indication(){
    return $('stems').hidden || $('mix-settings').disabled ||
      $('solo-target').disabled || (!documentlessHidden('stale-separation')) ||
      ($('notice').textContent!==was && /更改|重新|重做|先前|上次|舊/.test($('notice').textContent));
  }
  function documentlessHidden(id){return !$(id)||$(id).hidden;}
  record('stale-after-target-reference-edit',indication(),{retainedTarget:$('dl-target').href===stemURL,soloEnabled:!$('solo-target').disabled,renderEnabled:!$('mix-settings').disabled,downloadVisible:!$('exports').hidden,statusMessage:$('notice').textContent});
  $('mode').value='rnnoise';$('mode').onchange();
  record('stale-after-method-change',indication(),{selected:$('mode').value,retainedTarget:$('dl-target').href===stemURL,soloEnabled:!$('solo-target').disabled,renderEnabled:!$('mix-settings').disabled,downloadVisible:!$('exports').hidden,statusMessage:$('notice').textContent});
  await $('separate').onclick();
  record('new-separation-clears-prior-render',$('dl-target').href!==stemURL&&$('exports').hidden&&$('listen-mix').disabled);
  $('output-mode').value='headphones';$('output-mode').onchange();$('pan').value='45';fire('pan','input');await $('render').onclick();
  record('headphone-render',$('pan-val').textContent==='+45°'&&!$('exports').hidden&&!$('notice').classList.contains('error'));
  await $('demo').onclick();
  record('switch-clip-clears-old-stems-and-mix',$('stems').hidden&&$('exports').hidden&&$('mix-settings').disabled&&audio.src.endsWith('/original.wav'));
  const caps=await (await fetch(new URL('/api/capabilities',base))).json();
  report.capabilities=caps.modes.map(m=>m.id);
  fs.writeFileSync(path.join(tmp,'dom.json'),JSON.stringify(report,null,2));
}
main().catch(err=>{console.error(err);process.exitCode=1;});
