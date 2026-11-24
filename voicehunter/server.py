"""Loopback-only workbench. All temporary audio lives in a private per-run folder."""
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import RLock
import uuid
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
from .engine import modes, separate_clip, mix_stems, read_span
from .media import load_media, waveform, write_wav, mux_video, MAX_UPLOAD, MAX_SECONDS, SAMPLE_RATE, EXTENSIONS

BASE = Path(__file__).resolve().parent
WEB = BASE / "web"


def new_id():
    return uuid.uuid4().hex[:16]


class Session:
    def __init__(self, root, iphone):
        self.root = Path(root)
        self.iphone = iphone
        self.clips = OrderedDict()
        self.lock = RLock()

    def import_clip(self, raw_path, name, key=None, hints=None):
        clip_id = key or new_id()
        directory = self.root / clip_id
        if clip_id in self.clips:
            return self.clips[clip_id]["info"]
        x, info = load_media(raw_path)
        directory.mkdir(mode=0o700)
        try:
            src = directory / ("source" + raw_path.suffix.lower())
            shutil.copyfile(raw_path, src)
            np.save(directory / "original.npy", x)
            write_wav(directory / "original.wav", x)
            info.update(id=clip_id, name=name, duration=len(x)/SAMPLE_RATE, sample_rate=SAMPLE_RATE,
                        channels=2, peaks=waveform(x), original_url=f"/media/{clip_id}/original.wav")
            info.update(hints or {})
            self.clips[clip_id] = {"info":info,"source":src,"seps":OrderedDict(),"renders":OrderedDict()}
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        while len(self.clips) > 5:
            old, _ = self.clips.popitem(last=False)
            shutil.rmtree(self.root / old, ignore_errors=True)
        return info

    def get(self, clip_id):
        if not isinstance(clip_id, str) or clip_id not in self.clips:
            raise ValueError("找不到這段素材，請重新匯入。")
        self.clips.move_to_end(clip_id)
        return self.clips[clip_id], self.root / clip_id

    def separate(self,p):
        clip_id = p.get("id")
        clip,d = self.get(clip_id)
        mode = str(p.get("mode","example"))
        x = np.load(d / "original.npy", mmap_mode="r")
        span = bg = None
        if mode in ("example","speakers","speakers_moving","spatial-example"):
            span = read_span(p,"start","end")
            bg = read_span(p,"background_start","background_end")
            if not span or (mode in {"speakers","speakers_moving","spatial-example"} and not bg):
                raise ValueError("請標記目標示範段。雙聲源模式還需要第二個聲源的單獨示範段。")
            if any(s is not None and s[1] > len(x)/SAMPLE_RATE + .001 for s in (span,bg)):
                raise ValueError("示範段超出音訊長度。")
            if bg and min(bg[1],span[1]) > max(bg[0],span[0]):
                raise ValueError("兩段示範時間不能重疊。")
        t,b,meta = separate_clip(x, mode, str(p.get("target") or "")[:180], span,bg)
        sid = new_id()
        prefix = f"/media/{clip_id}/{sid}"
        for label, data in (("target",t),("residual",b)):
            write_wav(d / f"{sid}-{label}.wav", data, stem=True)
        clip["seps"][sid] = True
        while len(clip["seps"]) > 4:
            stale,_ = clip["seps"].popitem(last=False)
            for f in d.glob(f"{stale}-*.wav"):
                f.unlink(missing_ok=True)
        return {"id":clip_id,"separation_id":sid,"target_url":prefix+"-target.wav",
                "residual_url":prefix+"-residual.wav","note":meta.get("note") or meta.get("description",""),
                "method":mode,"metadata":meta}

    def render(self,p):
        clip_id = p.get("id")
        clip,d = self.get(clip_id)
        sid = p.get("separation_id")
        if sid not in clip["seps"]:
            raise ValueError("請重新分軌，再調整混音。")
        # Our float WAV writers put the samples last. Use the exact source length.
        n = int(np.load(d / "original.npy", mmap_mode="r").shape[0])
        def load_stem(label):
            f = d / f"{sid}-{label}.wav"
            return np.memmap(f, mode="r", dtype="<f4", offset=f.stat().st_size - n*8, shape=(n,2))
        mixed = mix_stems(load_stem("target"),load_stem("residual"),
                          float(p.get("gain_db",0)),float(p.get("pan",0)),float(p.get("width",1)),
                          output_mode=p.get("output_mode","stereo"),azimuth_deg=float(p.get("azimuth_deg",0)))
        rid = new_id()
        g = write_wav(d / f"{rid}-mix.wav", mixed)
        clip["renders"][rid] = sid
        while len(clip["renders"]) > 6:
            old,_ = clip["renders"].popitem(last=False)
            for f in d.glob(f"{old}-*"):
                f.unlink(missing_ok=True)
        return {"render_id":rid,"mix_url":f"/media/{clip_id}/{rid}-mix.wav",
                "peak_gain_db":round(float(20*np.log10(g)),3),"has_video":clip["info"]["has_video"]}

    def video(self,p):
        clip_id = p.get("id")
        clip,d = self.get(clip_id)
        rid = p.get("render_id")
        if not clip["info"]["has_video"] or rid not in clip["renders"]:
            raise ValueError("請先為這段影片產生混音。")
        path = d / f"{rid}-video.mov"
        if not path.exists():
            tmp = path.with_suffix(".part")
            try:
                mux_video(clip["source"], d / f"{rid}-mix.wav", tmp)
                os.replace(tmp,path)
            finally:
                tmp.unlink(missing_ok=True)
        return {"video_url":f"/media/{clip_id}/{path.name}"}


class Service(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, data, status=200):
        content = json.dumps(data,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v),allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Cache-Control","no-store")
        self.send_header("Content-Length",str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def local(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}",f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host","").lower() not in hosts or (origin and origin not in {f"http://{h}" for h in hosts}):
            self.send_json({"error":"僅限本機頁面使用。"},403)
            return False
        return True

    def serve_file(self,path):
        if not path.is_file():
            self.send_error(404)
            return
        with path.open("rb") as f:
            size = os.fstat(f.fileno()).st_size
            start, end = 0, size-1
            rng = self.headers.get("Range")
            if rng:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)",rng)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                a,b=match.groups()
                if a:
                    start=int(a)
                    if b: end=min(end,int(b))
                else:
                    start=max(0,size-int(b))
                if start>end:
                    self.send_error(416)
                    return
                self.send_response(206)
                self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length",str(end-start+1))
            self.send_header("Accept-Ranges","bytes")
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            if self.command=="HEAD": return
            f.seek(start)
            remaining=end-start+1
            while remaining:
                block=f.read(min(remaining,262144))
                if not block: break
                self.wfile.write(block)
                remaining-=len(block)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.local(): return
        parsed=urlsplit(self.path)
        route=parsed.path
        s=self.server.session
        try:
            if route == "/api/capabilities":
                self.send_json({"modes":modes(),"headphones":True,"iphone_demo":bool(s.iphone),"max_seconds":MAX_SECONDS})
            elif route=="/api/demo":
                with s.lock:
                    choice = parse_qs(parsed.query).get("clip", ["studio"])[0]
                    demos = {"studio": (BASE.parent/"examples"/"studio.wav", "a"*16,
                                {"mode":"example","start":0.05,"end":2.05,"background_start":2.35,"background_end":4.30}),
                             "duet": (BASE.parent/"examples"/"duet.wav", "c"*16,
                                {"mode":"speakers","start":0.2,"end":1.65,"background_start":2.2,"background_end":3.65,
                                 "note":"兩個不同位置的合成人聲型音色，先各自示範，再重疊發聲。"}),
                             "iphone": (s.iphone, "b"*16, {"mode":"rnnoise",
                                "note":"室內拍電視的原始音軌。可先提取人聲，再比較混音。"})}
                    if choice not in demos or not demos[choice][0]:
                        raise ValueError("這個示範素材未載入。")
                    path, key, hints = demos[choice]
                    self.send_json(s.import_clip(path, path.name, key, hints))
            elif route in ("/","/index.html","/style.css","/app.js"):
                self.serve_file(WEB/("index.html" if route=="/" else route[1:]))
            elif re.fullmatch(r"/media/[a-f0-9]{16}/(?:original\.wav|[a-f0-9]{16}-(?:target\.wav|residual\.wav|mix\.wav|video\.mov))",route):
                self.serve_file(s.root/route[len("/media/"):])
            else:
                self.send_error(404)
        except (BrokenPipeError,ConnectionResetError): pass
        except (ValueError,OSError,RuntimeError) as e:
            self.send_json({"error":str(e)},400)

    def do_POST(self):
        if not self.local(): return
        s=self.server.session
        route=urlsplit(self.path).path
        try:
            n=int(self.headers.get("Content-Length","0"))
            if route=="/api/import":
                from tempfile import NamedTemporaryFile
                raw_name=unquote(self.headers.get("X-Filename","clip.wav"))
                name=Path(raw_name.replace(chr(92),"/")).name[:180]
                suffix=Path(name).suffix.lower()
                if suffix not in EXTENSIONS: raise ValueError("請選擇支援的影片或音訊格式。")
                if not 0<n<=MAX_UPLOAD: raise ValueError("請選擇 160 MiB 以內的檔案。")
                with NamedTemporaryFile(dir=s.root,suffix=suffix) as tmp:
                    left=n
                    while left:
                        chunk=self.rfile.read(min(left,262144))
                        if not chunk: raise ValueError("上傳中斷。")
                        tmp.write(chunk);left-=len(chunk)
                    tmp.flush()
                    with s.lock:
                        result=s.import_clip(Path(tmp.name),name)
            elif route in ("/api/separate","/api/render","/api/video"):
                if not 0<n<=16384: raise ValueError("參數長度不正確。")
                p=json.loads(self.rfile.read(n))
                if not isinstance(p,dict): raise ValueError("參數格式不正確。")
                with s.lock:
                    action={"/api/separate":s.separate,"/api/render":s.render,"/api/video":s.video}[route]
                    result=action(p)
            else:
                self.send_error(404);return
            self.send_json(result)
        except (BrokenPipeError,ConnectionResetError): pass
        except (ValueError,TypeError,KeyError,RuntimeError,OSError,ArithmeticError) as e:
            self.send_json({"error":str(e)},400)


def serve(port, iphone=None):
    with TemporaryDirectory(prefix="voicehunter-work-") as tmp:
        http=Service(("127.0.0.1",port), Handler)
        http.session=Session(tmp,iphone)
        print(f"Voice Hunter · http://127.0.0.1:{http.server_port}/",flush=True)
        try: http.serve_forever()
        except KeyboardInterrupt: pass
        finally: http.server_close()
