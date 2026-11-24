"""Adapter for the validation-selected five-event research mode."""
from functools import lru_cache
from importlib.util import find_spec
import json
from pathlib import Path

from .media import SAMPLE_RATE
from .separation import auto_available

ROOT=Path(__file__).resolve().parent.parent
CHOICES=(("speech","人聲"),("piano","鋼琴"),("guitar","吉他"),("train","列車"),("wind","環境風聲"))


def selected():
    lock=ROOT/'round3/evaluation/selection.json'
    if lock.is_file():
        return json.loads(lock.read_text())['primary']
    return None


def available():
    name=selected()
    if name in ('fusion','transfer','transformer','mixture') and find_spec('onnxruntime') is None:
        return False
    if name in ('fusion','apple_features') and not auto_available():
        return False
    return name in ('fusion','apple_features','transfer','transformer','classifier','mixture')


@lru_cache(maxsize=1)
def predictor():
    name=selected()
    if name=='fusion':
        from round3.fusion.model import separate
    elif name=='apple_features':
        from round3.apple_features.model import separate
    elif name=='classifier':
        from round3.classifier.model import separate
    elif name=='mixture':
        from round3.mixture.model import separate
    elif name=='transformer':
        from round3.transformer.model import predict_scene
        def separate(audio,sr,target):
            return predict_scene(audio,sr).get(target)
    elif name=='transfer':
        from round3.transfer.candidate import Predictor
        backend=Predictor()
        def separate(audio,sr,target):
            return backend.prepare(audio,sr).predict(target)
    else:
        raise ValueError('找不到已通過訓練與驗證的研究模型。')
    return separate


def extract(audio,target):
    if target not in dict(CHOICES):raise ValueError('請選擇五種支援的聲音之一。')
    if not available():raise ValueError('請依 README 安裝研究模式所需的執行環境。')
    stem=predictor()(audio,SAMPLE_RATE,target)
    if isinstance(stem,tuple):
        stem=stem[0]
    if stem is None:raise ValueError('兩個聲源的方位或類別不夠明確，請改用示範段。')
    return stem, audio-stem, {'method':'five-events','model':selected(),'target':target,
        'note':'適用於兩個分開且固定的聲像。先獨聽目標與其餘聲音。'}
