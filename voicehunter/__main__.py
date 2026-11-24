"""Voice Hunter local server and batch processor."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

from .engine import modes, separate_clip, mix_stems
from .media import SAMPLE_RATE, load_media, write_wav, mux_video


def main():
    p=argparse.ArgumentParser(description="Voice Hunter · Stereo source separation and remix")
    sub=p.add_subparsers(dest="command", required=True)
    web=sub.add_parser("serve", help="start local listening workbench")
    web.add_argument("--port",type=int,default=8845)
    web.add_argument("--demo-file",type=Path,help="make a local recording accessible as a second demo")
    sub.add_parser("modes",help="list available algorithms on this machine")
    work=sub.add_parser("process",help="export two aligned stems, a stereo mix, and settings")
    work.add_argument("input",type=Path)
    work.add_argument("--out",type=Path,required=True,help="new or empty output directory")
    work.add_argument("--mode",default="example")
    work.add_argument("--target",default=None)
    work.add_argument("--span",type=float,nargs=2,metavar=("START","END"))
    work.add_argument("--background",type=float,nargs=2,metavar=("START","END"))
    work.add_argument("--gain",type=float,default=0,help="foreground gain (dB, -12..12)")
    work.add_argument("--pan",type=float,default=0,help="relative shift (-1..1)")
    work.add_argument("--width",type=float,default=1,help="background width (0..2)")
    work.add_argument("--output-mode",choices=("stereo","headphones","headphones-measured"),default="stereo")
    work.add_argument("--azimuth",type=float,default=0,help="headphones only: horizontal angle, in degrees")
    work.add_argument("--video",action="store_true",help="also mux the new mix with the original picture")
    args=p.parse_args()
    try:
        if args.command=="serve":
            from .server import serve
            serve(args.port,args.demo_file)
        elif args.command=="modes":
            print(json.dumps(modes(),ensure_ascii=False,indent=2))
        else:
            if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
                raise ValueError("Select a new or empty output folder.")
            x,info=load_media(args.input)
            if args.video and not info["has_video"]:
                raise ValueError("This file has no video stream.")
            t,b,meta=separate_clip(x,args.mode,args.target or ("voice" if args.mode=="rnnoise" else "speech"),args.span,args.background)
            y=mix_stems(t,b,args.gain,args.pan,args.width,output_mode=args.output_mode,azimuth_deg=args.azimuth)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".voicehunter-", dir=args.out.parent) as temp:
                staged = Path(temp) / "export"
                staged.mkdir()
                write_wav(staged/"target.wav",t,stem=True)
                write_wav(staged/"remaining.wav",b,stem=True)
                g=write_wav(staged/"remix.wav",y)
                if args.video:
                    mux_video(args.input,staged/"remix.wav",staged/"remix.mov")
                meta.update(source=args.input.name, original=info, samples=len(x), sample_rate=SAMPLE_RATE,
                            render={"mode":args.output_mode,"azimuth_deg":args.azimuth,"gain_db":args.gain,"pan":args.pan,"width":args.width,"peak_protection_gain":g})
                (staged/"settings.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                staged.rename(args.out)
            print(args.out.resolve())
    except (ValueError,OSError,RuntimeError,ArithmeticError) as e:
        sys.exit(str(e))


if __name__=="__main__":
    main()
