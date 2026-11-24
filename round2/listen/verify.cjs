/* Verify embedded PCM and exercise the page controls using a simulated Web Audio clock. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const report = JSON.parse(fs.readFileSync(path.join(__dirname,'calibration.json'),'utf8'));
const material = html.match(/<script id="material" type="application\/json">(.*?)<\/script>/s)[1];
const js = html.match(/<script>\s*('use strict';.*?)<\/script>/s)[1];
const data = JSON.parse(material);
const checks=[];
for(const [name, encoded] of Object.entries(data)) {
  const bytes=Buffer.from(encoded,'base64');
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),report.asset[name].sha256);
  assert.equal(bytes.toString('ascii',0,4),'RIFF'); assert.equal(bytes.readUInt16LE(22),2);
  assert.equal(bytes.readUInt32LE(24),48000); assert.equal(bytes.readUInt16LE(34),16);
  assert.equal(bytes.readUInt32LE(40)/4,228000);
}
checks.push('five embedded stereo PCM hashes, rate, duration and headers');
class Button {constructor(dataset={}) {this.dataset=dataset;this.attrs={};this.textContent='';this.value=0;}
  setAttribute(k,v){this.attrs[k]=v;}}
async function trial(random) {
 const nodes=[], gainNodes=[], buttons={}, poses=['L','R'].map(side=>new Button({side})),
 conds=['ref','a','b'].map(cond=>new Button({cond}));
 for(const id of ['material','play','answer','status','position','reveal'])buttons[id]=new Button();
 buttons.material.textContent=material;
 let clock;
 class Context {
  constructor(){this.currentTime=0;this.destination={};clock=this;}
  async resume(){} async decodeAudioData(){return {duration:4.75};}
  createBufferSource(){const node={connect(){},disconnect(){},start(t,offset){this.when=t;this.offset=offset;},stop(){this.stopped=true;}};nodes.push(node);return node;}
  createGain(){const g={connect(){},disconnect(){},gain:{setTargetAtTime(x){this.value=x;}}};gainNodes.push(g);return g;}
 }
 const context={window:{AudioContext:Context,addEventListener(){}},document:{querySelector(q){return buttons[q.slice(1)];},
     querySelectorAll(q){return q==='.pos'?poses:conds;}},atob:x=>Buffer.from(x,'base64').toString('binary'),
     crypto:{getRandomValues:a=>{a[0]=random;return a;}},Uint8Array,requestAnimationFrame(){}};
 vm.runInNewContext(js,context);
 await buttons.play.onclick();assert.equal(nodes.length,5);
 assert.equal(new Set(nodes.map(v=>v.when)).size,1);
 assert.equal(new Set(nodes.map(v=>v.offset)).size,1);
 const active=()=>Object.keys(data)[gainNodes.slice(-5).findIndex(g=>g.gain.value===1)];
 assert.equal(active(),'ref');
 clock.currentTime=.9;conds[1].onclick();assert.equal(active(),(random?'measured':'analytic')+'R');
 poses[0].onclick();assert.equal(active(),(random?'measured':'analytic')+'L');
 assert.equal(nodes.length,5);assert.equal(buttons.answer.textContent,'');
 buttons.position.value=500;await buttons.position.onchange();
 assert.equal(nodes.length,10);assert.equal(nodes[5].offset,2.375);
 buttons.reveal.onclick();assert.ok(buttons.answer.textContent.includes('CIPIC'));
 await buttons.play.onclick();assert.equal(buttons.play.attrs['aria-pressed'],'false');
}
(async()=>{
 await trial(0);await trial(1);
 checks.push('both coded orders, mirror selection, initial sync, switching with same playhead, midclip seek, pause, deferred reveal');
 fs.writeFileSync(path.join(__dirname,'verification.json'),JSON.stringify({checks,evidence:'DOM/Web Audio clock simulation; no human or physical-device playback'},null,2)+'\n');
 console.log('Listening-page checks passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
