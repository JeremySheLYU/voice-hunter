"""One interface for import, source estimation and stereo rendering."""
import numpy as np
from .media import SAMPLE_RATE
from .spatial import remix, mask_from_estimates, extract_stereo


def modes():
    items = [{"id": "example", "label": "選一段聲音作示範", "kind": "example", "note": "用示範段學習聲音的頻譜，再從整段錄音提取相似聲音。"},
             {"id": "speakers", "label": "雙人對話 · 固定位置", "kind": "example",
              "requires_background": True, "note": "各選 A、B 至少 0.35 秒獨講，適合兩人位置相對固定的對話。"}]
    items.append({"id": "speakers_moving", "label": "雙人對話 · 移動追蹤（實驗）", "kind": "example",
                  "requires_background": True, "note": "用兩段獨講示範和音色追蹤位置互換。強反射可能無法分離，請試聽確認身分。"})
    items.append({"id": "spatial-example", "label": "雙聲源 · 空間與音色分軌", "kind": "example",
                  "requires_background": True, "note": "適合位置固定的兩個聲源。各選一段清楚示範；聲源換位或要保留底聲時可用原有示範模式。"})
    from .separation import auto_available
    if auto_available():
        items.append({"id": "auto", "label": "自動尋找聲音 · Apple 辨識", "kind": "choice", "note": "先辨識目標出現的時段，再估計音軌。片段中若始終有同一種聲音，請改用示範段。",
                      "options": [{"value": k, "label": v} for k, v in [
                          ("speech", "說話聲"), ("piano", "鋼琴"), ("guitar", "吉他"),
                          ("music", "音樂"), ("train", "列車"), ("wind", "環境風聲"),
                          ("wind_mic", "風吹麥克風")]]})
    from .five_events import available, CHOICES
    if available():
        items.append({"id":"five-events", "label":"五類自動分軌 · 固定聲像（實驗）", "kind":"choice",
                      "note":"從人聲、吉他、鋼琴、列車、環境風聲選一類。適合兩個位置穩定的聲源。",
                      "options":[{"value":key,"label":label} for key,label in CHOICES]})
    from .pretrained import capabilities
    items.extend(m for m in capabilities() if m.get("available", False))
    return items


def separate_clip(x, mode, target="", span=None, background=None):
    if mode == "speakers":
        from .speech import separate_speakers
        t, b, meta = separate_speakers(x, SAMPLE_RATE, target_span=span, background_span=background)
    elif mode == "speakers_moving":
        from .speech import separate_moving_speakers
        t, b, meta = separate_moving_speakers(x, SAMPLE_RATE, target_span=span, background_span=background)
    elif mode == "spatial-example":
        if span is None or background is None:
            raise ValueError("請各選一段目標和第二聲源的示範音訊。")
        from .multitrack import separate_spatial_examples
        t, b, meta = separate_spatial_examples(x, span, background)
    elif mode == "five-events":
        from .five_events import extract
        t, b, meta = extract(x,target)
    elif mode in {"example", "auto"}:
        from .separation import separate
        t, b, meta = separate(x, SAMPLE_RATE, method=mode, target=target or "speech",
                              target_span=span, background_span=background)
    else:
        from .pretrained import capabilities, separate_pretrained
        enabled = {m["id"]:m for m in capabilities() if m.get("available", False)}
        if mode not in enabled:
            raise ValueError("目前這台電腦沒有啟用這個模型。")
        t, b = separate_pretrained(x, SAMPLE_RATE, mode, target)
        if np.shape(t) in [(len(x),), (len(x), 1)]:
            mask = mask_from_estimates(t.ravel(), b.ravel())
            t, b = extract_stereo(x, mask)
        elif np.shape(t) == x.shape:
            b = x - t
        meta = {"method": mode, "note": enabled[mode]["label"]}
    t, b = np.asarray(t, np.float32), np.asarray(b, np.float32)
    if (t.shape != x.shape or b.shape != x.shape or not np.isfinite(t).all()
            or not np.isfinite(b).all()):
        raise ValueError("分軌輸出不完整，請試用另一種模式。")
    return t, b, meta


def mix_stems(t, b, gain, pan, width, *, output_mode="stereo", azimuth_deg=0):
    if not (-12 <= gain <= 12 and -1 <= pan <= 1 and 0 <= width <= 2):
        raise ValueError("參數超出範圍。")
    if output_mode in {"headphones", "headphones-measured"}:
        from .binaural import render_headphones
        return render_headphones(t, b, SAMPLE_RATE, azimuth_deg=azimuth_deg,
                                 width=width, target_gain_db=gain,
                                 profile="measured" if output_mode == "headphones-measured" else "analytic")
    if output_mode != "stereo":
        raise ValueError("未知的混音模式。")
    return remix(t, b, target_gain_db=gain, pan=pan, width=width)


def read_span(params, k1, k2):
    a,b = params.get(k1), params.get(k2)
    if a is None and b is None:
        return None
    if a is None or b is None or not (np.isfinite(float(a)) and np.isfinite(float(b)) and 0 <= float(a) < float(b)):
        raise ValueError("請輸入有效的起點和終點（秒）。")
    return float(a), float(b)
