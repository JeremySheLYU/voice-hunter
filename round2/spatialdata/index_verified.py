"""Verify file-level properties and SHA-256 for actually present spatial assets."""
import hashlib
import json
from pathlib import Path
import subprocess
import wave

HERE=Path(__file__).resolve().parent
RAW=HERE.parents[1]/'round2_data/spatial'


def read(name):
    return json.loads((HERE/name).read_text())


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def summarize():
    satp=read('satp_index.json')
    aachen=read('aachen_index.json')
    sources=read('binaurec_plan.json')['scenes']
    referred=[]
    for s in satp+aachen['paired_binaural_or_headless']:
        f=RAW/s['file']
        assert digest(f)==s['sha256'],s['file']
        referred.append(f)
    bins=[]
    for s in sources:
        f=RAW/'binaurec'/s['scene']
        checks=json.loads((f/'hashes.json').read_text());records={}
        for k,h in checks.items():
            fp=f/(k+'.wav')
            assert digest(fp)==h
            with wave.open(str(fp)) as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,2,2)
                records[k]=dict(sha256=h,channels=2,sr=16000,frames=w.getnframes(),
                                file=str(fp.relative_to(RAW)))
            referred.append(fp)
        bins.append(dict(scene=s['scene'],utterance_id=s['utterance_id'],paths=records))
    apl=[]
    for row in read('apl_plan.json')['members']:
        name,length,*_=row
        f=RAW/'apl'/Path(name).name
        assert f.is_file() and f.stat().st_size==length,f
        rec=dict(file=str(f.relative_to(RAW)),bytes=length,sha256=digest(f),split='external-room-reserve',
                 venue='St Paul’s concert hall, Huddersfield',member=name)
        if f.suffix=='.wav':
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(f)]))['streams'][0]
            assert int(info['channels'])==4
            rec.update(format='AmbiX FOA (ACN/SN3D)',sr=int(info['sample_rate']),channels=4,
                       duration=float(info['duration']),codec=info['codec_name'])
        else:
            import h5py
            with h5py.File(f) as h:
                shape=list(h['Data.IR'].shape)
                rec.update(format='SOFA KU100 BRIR',sr=int(h['Data.SamplingRate'][0]),
                           shape=shape,sofa_convention=h.attrs['SOFAConventions'].decode(),
                           data_delay_shape=list(h['Data.Delay'].shape))
                assert shape[1]==2 and shape[0]>=95
        apl.append(rec);referred.append(f)
    duration=sum(r['frames']/r['sr'] for r in satp if r['type']=='binaural_field')
    irrows=aachen['paired_binaural_or_headless']
    report=dict(satp_binaural_captures=27,satp_seconds=duration,satp_calibration_signals=1,
                aachen_verified_interleaved_pairs=len(irrows),aachen_binaural_head_pairs=sum(i['head'] for i in irrows),
                aachen_headless_pairs=sum(not i['head'] for i in irrows),aachen_unpaired_phone_mono_members=len(aachen['other_phone_single_channel_wavs']),
                apl_foa_room_responses=len([r for r in apl if r['format'].startswith('AmbiX')]),apl_brir_sofa=[r for r in apl if r['format'].startswith('SOFA')],
                binaurec_unique_ids=len(bins),binaurec_recorded_programme_seconds=sum(r['paths']['mixture_ssn']['frames']/16000 for r in bins),
                binaurec_paired_signals=len(bins)*3,audited_local_files=len(referred),audited_file_bytes=sum(p.stat().st_size for p in referred))
    (HERE/'spatial_validation.json').write_text(json.dumps(dict(summary=report,apl=apl,binaurec=bins),indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':summarize()
