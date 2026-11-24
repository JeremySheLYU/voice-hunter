"""Pick an available local port, open the browser only after bind succeeds."""
from pathlib import Path
import sys
import webbrowser
import threading
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicehunter.server import Service, Session, Handler

with tempfile.TemporaryDirectory(prefix="voicehunter-work-") as temp:
    try:
        http = Service(("127.0.0.1",8845), Handler)
    except OSError:
        http = Service(("127.0.0.1",0), Handler)
    http.session = Session(temp,None)
    url = f"http://127.0.0.1:{http.server_port}/?demo=studio"
    print(f"Voice Hunter · {url}\n按 Ctrl+C 關閉。",flush=True)
    threading.Timer(.2,webbrowser.open,args=(url,)).start()
    try: http.serve_forever()
    except KeyboardInterrupt: pass
    finally: http.server_close()
