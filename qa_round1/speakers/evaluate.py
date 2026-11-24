"""Frozen v0.1 stereo separation stress test. Run from any directory.

python3 qa_round1/speakers/evaluate.py --cache /tmp/vh-fsdd --fetch
python3 qa_round1/speakers/evaluate.py --outfile /tmp/synthetic-results.json
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import urllib.request
import wave

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from voicehunter.speech import separate_speakers

FS = 48000
DURATION = 8.0
A_REF, B_REF = (0.14, 1.06), (1.54, 2.46)
A_SOLO, B_SOLO, OVERLAP = (3.12, 3.87), (4.42, 5.17), (5.92, 7.56)
PAIRS = [('jackson', 'nicolas'), ('theo', 'yweweler'), ('lucas', 'george')]
SCENARIOS = ['fixed', 'reversed', 'weak_target', 'half_overlap', 'near',
             'coincident', 'strong_reflections', 'crossing', 'contaminated']
EPS = 1.e-20


def hashes():
    return json.loads((HERE / 'source_manifest.json').read_text())


def samples(cache, fetch):
    manifest = hashes()
    missing = []
    for name, expected in manifest['files'].items():
        path = cache / name if cache else None
        if path and fetch and not path.exists():
            url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['commit']}/recordings/{name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if sha256(data).hexdigest() != expected:
                raise ValueError(f'Wrong download digest: {name}')
            path.write_bytes(data)
        if not path or not path.exists():
            missing.append(name)
        elif sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Wrong local digest: {path}')
    return missing


def ramp(x, dur=0.012):
    n = min(len(x)//2, int(FS * dur))
    y = x.copy()
    if n:
        fade = np.sin(np.linspace(0, np.pi/2, n))**2
        y[:n] *= fade
        y[-n:] *= fade[::-1]
    return y


def human(cache, who):
    clips = []
    for d in range(8):
        with wave.open(str(cache / f'{d}_{who}_{10 if d < 4 else 2}.wav'), 'rb') as inp:
            assert inp.getnchannels() == 1 and inp.getsampwidth() == 2
            source_fs = inp.getframerate()
            mono = np.frombuffer(inp.readframes(inp.getnframes()), dtype='<i2') / 32768.
        new_n = round(len(mono) * FS / source_fs)
        resampled = np.interp(np.arange(new_n) * source_fs / FS, np.arange(len(mono)), mono)
        clips.append(ramp(resampled))
    return clips


def artificial(seed):
    rng = np.random.default_rng(seed)
    clips = []
    identity_f0 = rng.uniform(112, 203)
    tract = rng.uniform(.91, 1.15)
    vowels = np.array([[320,2190,2830], [700,1270,2450],
                       [530,1770,2540], [340,875,2110], [450,1450,2680]])
    for _ in range(8):
        pieces = []
        for __ in range(2):
            dur = rng.uniform(.14, .24)
            t = np.arange(int(dur * FS)) / FS
            f0 = identity_f0 * rng.uniform(.9, 1.08) * (1+.085*np.sin(2*np.pi*t/dur))
            phase = 2 * np.pi * np.cumsum(f0) / FS
            centers = tract * vowels[rng.integers(len(vowels))]
            harmonics = np.arange(1, 42)[:,None]
            freq = harmonics * f0[None,:]
            profile = sum(g/(1+((freq-f)/w)**2) for f,w,g in zip(centers,[65,105,165],[1,.65,.40]))
            voiced = np.sum(np.sin(harmonics*phase) * (profile+.025)/np.sqrt(harmonics), axis=0)
            fricative = np.diff(rng.normal(size=round(FS*.038)), prepend=0) * .055
            pieces.extend([ramp(fricative), ramp(voiced), np.zeros(round(FS*.025))])
        clips.append(np.concatenate(pieces))
    return clips


def crop_cycle(parts, seconds):
    wavelets = np.concatenate([np.r_[p, np.zeros(round(.035*FS))] for p in parts])
    n = round(seconds * FS)
    x = ramp(np.resize(wavelets, n))
    return x / max(np.sqrt(np.mean(x*x)), .0001) * .06


def put(track, start, sound, gain=1.):
    i = round(start * FS)
    track[i:i+len(sound)] += gain*sound


def solo_tracks(a, b, case):
    x = np.zeros(round(DURATION * FS))
    y = np.zeros_like(x)
    ca, cb = crop_cycle(a[:4],1.12), crop_cycle(b[:4],1.12)
    put(x,.04,ca)
    put(y,1.44,cb)
    if case=='contaminated':
        put(y,.04,cb,10**(-8/20))
        put(x,1.44,ca,10**(-8/20))
    for signal, src, start in ((x,a,3.0),(y,b,4.3)):
        put(signal,start,crop_cycle(src[4:6],1.0))
    ax = crop_cycle(a[6:],1.90)
    by = crop_cycle(b[6:][::-1],1.90)
    if case=='half_overlap':
        # Toggle B every 160 ms inside evaluation; A remains active.
        for i in range(0,len(by),int(.32*FS)):
            by[i:i+round(.16*FS)] = 0
    put(x,5.80,ax)
    put(y,5.80,by)
    if case=='weak_target':
        x *= 10**(-8/20)
    return x,y


def ir(pan, seed, strength):
    """Fixed synthetic early taps and diffuse tail. One impulse response per channel."""
    rng = np.random.default_rng(seed)
    n = round(FS * (.255 if strength=='strong' else .034))
    h = np.zeros((n,2))
    theta = (pan+1)*np.pi/4
    gains = (np.cos(theta), np.sin(theta))
    base = 28
    itd = round(.00047*FS*pan)
    for c in range(2):
        h[base+(max(0,itd) if c==0 else max(0,-itd)), c] = gains[c]
        taps = rng.choice(np.arange(160,min(1680,n)), 6, replace=False)
        h[taps,c] += rng.normal(size=6) * (.21 if strength=='strong' else .024)
        decay = np.exp(-np.arange(n)/FS/(.066 if strength=='strong' else .009))
        tail = rng.normal(size=n) * decay
        tail[:120] = 0
        h[:,c] += (.68 if strength=='strong' else .028) * tail/np.sqrt(np.sum(tail*tail))
    return h


def echo(x, h):
    size = 1 << (len(x)+len(h)-2).bit_length()
    return np.fft.irfft(np.fft.rfft(x,n=size)[:,None]*np.fft.rfft(h,n=size,axis=0),n=size,axis=0)[:len(x)]


def rendered(x,y,case,seed):
    ap,bp=-.49,.44
    if case=='reversed':
        ap,bp=.44,-.49
    elif case=='near':
        ap,bp=-.045,.045
    elif case=='coincident':
        ap=bp=.11
    strength = 'strong' if case=='strong_reflections' else 'weak'
    ha = ir(ap,seed,strength)
    hb = ha if case=='coincident' else ir(bp,seed+11,strength)
    ax,by = echo(x,ha),echo(y,hb)
    if case=='crossing':
        # Smoothly exchange whole acoustic paths; listener and calibration stay fixed.
        f = np.clip((np.arange(len(x))/FS-5.68)/.51,0,1)[:,None]
        ax = (1-f)*ax + f*echo(x,hb)
        by = (1-f)*by + f*echo(y,ha)
    rng = np.random.default_rng(seed+222)
    room = rng.normal(0,.00018,(len(x),2))
    return ax, by, room


def selection(pair):
    return slice(round(pair[0]*FS),round(pair[1]*FS))


def actives(a,b):
    # A joint overlap mask excludes true pauses. Threshold is per reference
    # peak frame power, calculated only within the held-out overlap segment.
    size=round(.020*FS)
    n=min(len(a),len(b))//size*size
    av=np.mean(a[:n].reshape(-1,size,2)**2,axis=(1,2))
    bv=np.mean(b[:n].reshape(-1,size,2)**2,axis=(1,2))
    mask=np.zeros(len(a),dtype=bool)
    mask[:n]=np.repeat((av>av.max()*1e-3)&(bv>bv.max()*1e-3),size)
    if mask.sum() < .10*FS:
        raise RuntimeError('Less than 100ms active overlap; fixture cannot be scored.')
    return mask


def powr(x):
    return float(np.mean(np.square(x.astype(float))))


def db_ratio(a,b):
    return float(10*np.log10((a+EPS)/(b+EPS)))


def snr(pred,ref):
    return db_ratio(powr(ref),powr(pred-ref))


def run(a,b,case,seed):
    x,y = solo_tracks(a,b,case)
    ta,tb,room = rendered(x,y,case,seed)
    mix=(ta+tb+room).astype('float32')
    res_ref=tb+room
    seg=selection(OVERLAP)
    mask=actives(ta[seg],tb[seg])
    aq=ta[seg][mask]
    bq=res_ref[seg][mask]
    mq=mix[seg][mask]
    basea,baseb=snr(mq,aq),snr(mq,bq)
    sa,sb=selection(A_REF),selection(B_REF)
    entry={
        'scenario':case, 'overlap_active_s':round(mask.sum()/FS,3),
        'A_example_other_vs_A_db':round(db_ratio(powr(tb[sa]+room[sa]),powr(ta[sa])),3),
        'B_example_other_vs_B_db':round(db_ratio(powr(ta[sb]+room[sb]),powr(tb[sb])),3),
        'input_snr_A_db':round(basea,3), 'input_snr_Broom_db':round(baseb,3),
    }
    try:
        out,rest,info=separate_speakers(mix,FS,target_span=A_REF,background_span=B_REF)
    except ValueError as exc:
        entry.update(status='refused',reason=str(exc))
        return entry
    a_snr=snr(out[seg][mask],aq)
    b_snr=snr(rest[seg][mask],bq)
    a_delta,b_delta=a_snr-basea,b_snr-baseb
    b_solo,a_solo=selection(B_SOLO),selection(A_SOLO)
    peakerr=float(np.max(abs(out+rest-mix)))
    entry.update(status='returned',gain_A_db=round(a_delta,3),gain_Broom_db=round(b_delta,3),
                 output_snr_A_db=round(a_snr,3),output_snr_Broom_db=round(b_snr,3),
                 min_gain_db=round(min(a_delta,b_delta),3),
                 B_only_rejection_db=round(db_ratio(powr(mix[b_solo]),powr(out[b_solo])),3),
                 A_only_level_error_db=round(db_ratio(powr(out[a_solo]),powr(ta[a_solo])),3),
                 neutral_max_abs_error=peakerr,finite=bool(np.isfinite(out).all() and np.isfinite(rest).all()),
                 calibration_spatial_contrast=info['spatial_contrast'],calibration_coherence=info['reference_coherence'])
    return entry


def summarize(entries):
    returned=[row for row in entries if row['status']=='returned']
    gains=[row['min_gain_db'] for row in returned]
    base={'total':len(entries),'returned':len(returned),'refused':sum(row['status']=='refused' for row in entries),
          'skipped':sum(row['status']=='skipped' for row in entries),'harmful_returned':sum(g<=0 for g in gains)}
    if gains:
        base.update(median_min_gain_db=round(float(np.median(gains)),3),worst_min_gain_db=min(gains),
                    median_B_only_rejection_db=round(float(np.median([row['B_only_rejection_db'] for row in returned])),3),
                    worst_reconstruction_error=max(row['neutral_max_abs_error'] for row in returned))
    return base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--outfile',type=Path,default=HERE/'results.json')
    args=parser.parse_args()
    if args.fetch and not args.cache:
        parser.error('--fetch requires --cache PATH')
    missing=samples(args.cache,args.fetch)
    runs=[]
    for corpus in ['synthetic','FSDD']:
        for i,(ta,tb) in enumerate(PAIRS):
            pair=f'{ta}/{tb}' if corpus=='FSDD' else f'synthetic-{i+1}'
            needed=[name for name in missing if name.split('_')[1] in (ta,tb)]
            if corpus=='FSDD' and needed:
                for case in SCENARIOS:
                    runs.append(dict(corpus=corpus,pair=pair,scenario=case,status='skipped',reason='public source WAVs absent: '+', '.join(needed)))
                continue
            a,b=(human(args.cache,ta),human(args.cache,tb)) if corpus=='FSDD' else (artificial(410+i*2),artificial(411+i*2))
            for scenario in SCENARIOS:
                run_result=run(a,b,scenario,50+i*31)
                runs.append(dict(corpus=corpus,pair=pair,**run_result))
    report={}
    for corpus in ('synthetic','FSDD'):
        rows=[r for r in runs if r['corpus']==corpus]
        report[corpus]={case:summarize([r for r in rows if r['scenario']==case]) for case in SCENARIOS}
    scoped=[r for r in runs if r['scenario'] in SCENARIOS[:4] and r['status']!='skipped']
    same=[r for r in runs if r['scenario']=='coincident' and r['status']!='skipped']
    sta=summarize(scoped)
    returned=[r for r in runs if r['status']=='returned']
    checks={
        'stationary_return_all_median_6dB_worst_3dB':bool(sta['returned']==len(scoped) and sta.get('median_min_gain_db',-999)>=6 and sta.get('worst_min_gain_db',-999)>=3),
        'coincident_all_refused':all(r['status']=='refused' for r in same),
        'finite_reconstruction_1e-5_all_returns':all(r['finite'] and r['neutral_max_abs_error']<1e-5 for r in returned),
    }
    payload={'method':'Frozen voicehunter.speech.separate_speakers', 'sample_rate':FS,
             'product_hashes':{f:sha256((HERE.parents[1]/'voicehunter'/f).read_bytes()).hexdigest() for f in ['speech.py','spatial.py']},
             'source_manifest_sha256':sha256((HERE/'source_manifest.json').read_bytes()).hexdigest(),
             'spans_s':dict(A_reference=A_REF,B_reference=B_REF,A_only=A_SOLO,B_only=B_SOLO,overlap=OVERLAP),
             'stationary_summary':sta,'overall':summarize(runs),'scenario_summary':report,'checks':checks,'runs':runs}
    args.outfile.parent.mkdir(parents=True,exist_ok=True)
    args.outfile.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:payload[k] for k in ['stationary_summary','overall','checks']},indent=2))
    for k,v in report.items():
        print(k,{s:(r['returned'],r['refused'],r.get('median_min_gain_db'),r.get('worst_min_gain_db')) for s,r in v.items()})


if __name__=='__main__':
    main()
