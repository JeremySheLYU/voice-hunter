/* Try genuine headless browser engines with isolated profiles. Does not attach to the user's tabs. */
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {VH_QA_BASE:base,VH_QA_DIR:tmp,VH_QA_VIDEO:video}=process.env;
const out={execution:'not-tested',attempts:[],checks:[]};
const save=()=>fs.writeFileSync(path.join(tmp,'browser.json'),JSON.stringify(out,null,2));
const cache=path.join(os.homedir(),'Library/Caches/ms-playwright');
function findLatest(prefix,suffix){
  if(!fs.existsSync(cache))return null;
  const choices=fs.readdirSync(cache).filter(x=>x.startsWith(prefix)).sort().reverse();
  return choices.map(x=>path.join(cache,x,suffix)).find(x=>fs.existsSync(x));
}
function reason(e){
  const lines=String(e).replace(/\x1b\[[0-9;]*m/g,'').split('\n');
  return (lines.find(x=>/bootstrap_check_in|FATAL:|Permission denied/i.test(x)) || lines.find(x=>/process did exit/.test(x)) || lines[0]).trim();
}
async function run(){
  let pw;
  for(const id of [process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core','/opt/homebrew/lib/node_modules/openclaw/node_modules/playwright-core'].filter(Boolean)) {try{pw=require(id);break;}catch{}}
  if(!pw){out.attempts.push({name:'Playwright',error:'Set PLAYWRIGHT_MODULE to a local installation'});save();return;}
  const options=[['system Chrome',pw.chromium,{channel:'chrome'}],
    ['Playwright Chromium',pw.chromium,{executablePath:findLatest('chromium_headless_shell-','chrome-headless-shell-mac-arm64/chrome-headless-shell')||undefined}],
    ['Playwright WebKit (not Safari)',pw.webkit,{executablePath:findLatest('webkit-','pw_run.sh')||undefined}]];
  let browser;
  for(const [name,engine,extra] of options){
    try {browser=await engine.launch({headless:true,timeout:7000,...extra});
      const page=await browser.newPage();await page.goto('about:blank');await page.close();out.execution=name;out.attempts.push({name,started:true});break;
    }catch(e){out.attempts.push({name,started:false,error:reason(e)});if(browser)await browser.close().catch(()=>{});browser=null;}
  }
  if(!browser){save();return;}
  try{
    for(const [name,width,height] of [['desktop',1280,800],['narrow viewport',375,812]]){
      const page=await browser.newPage({viewport:{width,height}});
      const errors=[];page.on('pageerror',e=>errors.push(String(e)));
      await page.goto(base+'?demo=studio');await page.waitForFunction(()=>document.querySelector('#filename').textContent==='studio.wav');
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      const buttons=await page.locator('button:visible').allTextContents();
      out.checks.push({id:name+' layout',horizontalOverflow:overflow,visibleButtons:buttons.length,passed:!overflow});
      const downloadPromise=async(selector)=>{const promise=page.waitForEvent('download');await page.click(selector);return (await promise).suggestedFilename();};
      await page.locator('#file').setInputFiles(video);
      await page.waitForFunction(()=>document.querySelector('#filename').textContent.endsWith('.mov'));
      await page.locator('#mode').selectOption('example');await page.locator('#start').fill('0.05');await page.locator('#end').fill('2.05');
      await page.click('#separate');await page.waitForSelector('#solo-target');
      await page.click('#solo-target');await page.waitForFunction(()=>!document.querySelector('#player').paused&&document.querySelector('#player').currentTime>0.05);
      await page.click('#solo-residual');await page.waitForFunction(()=>!document.querySelector('#player').paused&&document.querySelector('#player').currentTime>0.05);
      await page.click('#wide-mix');await page.waitForSelector('#dl-mix');
      await page.click('#listen-original');await page.click('#listen-mix');
      const mixName=await downloadPromise('#dl-mix');
      await page.click('#export-video');await page.waitForSelector('#dl-video');
      const movName=await downloadPromise('#dl-video');
      out.checks.push({id:name+' full workflow',mixName,movName,pageErrors:errors,passed:errors.length===0});
      await page.close();
    }
  }catch(e){out.checks.push({id:'browser workflow',passed:false,error:reason(e)});}
  finally{await browser.close();save();}
}
run().catch(e=>{out.attempts.push({name:'harness',error:reason(e)});save();});
