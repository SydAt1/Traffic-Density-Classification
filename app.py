import tempfile
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from torchvision import models, transforms

WEIGHTS_PATH = Path(__file__).parent / "models/efficientnet_b0_weights.pth"
CLASS_NAMES  = ["heavy", "incident", "light", "moderate"]

CLASS_COLORS_BGR = {
    "heavy":    (50,  57,  192),
    "incident": (0,   83,  211),
    "light":    (39,  174,  39),
    "moderate": (18,  156, 241),
}


# ── Preprocessing ─────────────────────────────────────────────────────────────
class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np     = np.array(img)
        img_lab    = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b    = cv2.split(img_lab)
        l_eq       = self.clahe.apply(l)
        img_lab_eq = cv2.merge((l_eq, a, b))
        return Image.fromarray(cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB))


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet-B0 model…")
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, len(CLASS_NAMES)),
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_frame(model, device, frame_bgr: np.ndarray) -> dict:
    img   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    t     = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)[0].cpu()
    idx = probs.argmax().item()
    return {
        "label":      CLASS_NAMES[idx],
        "confidence": probs[idx].item(),
        "probs":      {c: probs[i].item() for i, c in enumerate(CLASS_NAMES)},
    }


# ── Shared overlay drawing ────────────────────────────────────────────────────
def draw_overlay(img: np.ndarray, label: str, conf: float, probs: dict) -> np.ndarray:
    h, w      = img.shape[:2]
    color_bgr = CLASS_COLORS_BGR[label]

    # Dark top banner
    cv2.rectangle(img, (0, 0), (w, 58), (15, 15, 15), -1)

    # Coloured label pill
    pill_w = 9 + len(label) * 18
    cv2.rectangle(img, (8, 6), (8 + pill_w, 52), color_bgr, -1)
    cv2.putText(img, label.upper(), (16, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Confidence value
    cv2.putText(img, f"{conf * 100:.1f}%", (8 + pill_w + 10, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 220), 2, cv2.LINE_AA)

    # Mini probability bars (top-right)
    bar_x0 = w - 170
    bar_y  = 8
    for cls in CLASS_NAMES:
        p    = probs.get(cls, 0.0)
        bw   = int(p * 120)
        bcol = CLASS_COLORS_BGR[cls]
        cv2.rectangle(img, (bar_x0, bar_y), (bar_x0 + bw, bar_y + 9), bcol, -1)
        cv2.putText(img, f"{cls[:3]} {p*100:.0f}%",
                    (bar_x0 + 125, bar_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)
        bar_y += 13

    return img




# ── Live camera processor ─────────────────────────────────────────────────────
class TrafficVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._model, self._device = load_model()
        self._lock  = threading.Lock()
        self._label = "—"
        self._conf  = 0.0
        self._probs: dict[str, float] = {}
        self._tick  = 0

    @property
    def result(self) -> dict:
        with self._lock:
            return {"label": self._label, "confidence": self._conf, "probs": dict(self._probs)}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._tick += 1

        if self._tick % 4 == 0:
            pred = predict_frame(self._model, self._device, img)
            with self._lock:
                self._label = pred["label"]
                self._conf  = pred["confidence"]
                self._probs = pred["probs"]

        with self._lock:
            label = self._label
            conf  = self._conf
            probs = dict(self._probs)

        if label != "—":
            img = draw_overlay(img, label, conf, probs)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Traffic Analyser", layout="wide", page_icon="🚦")
st.title("🚦 Traffic Density Analyser")

model, device = load_model()
st.sidebar.success(f"Model ready · device: `{device}`")

tab_upload, tab_live = st.tabs(["📹 Video Upload", "📷 Live Camera"])


# ── Tab 1: Video Upload ───────────────────────────────────────────────────────
with tab_upload:
    st.caption("Upload a traffic video and watch it play with live predictions overlaid on screen.")

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"])
    run_btn  = st.button("Play", type="primary", disabled=uploaded is None)

    if uploaded and run_btn:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        cap   = cv2.VideoCapture(tmp_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_placeholder = st.empty()

        label, conf, probs = "—", 0.0, {}
        tick = 0

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            tick += 1
            if tick % 4 == 0:
                pred  = predict_frame(model, device, frame)
                label = pred["label"]
                conf  = pred["confidence"]
                probs = pred["probs"]

            if label != "—":
                frame = draw_overlay(frame, label, conf, probs)

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

            # Maintain original playback speed
            elapsed = time.time() - t0
            time.sleep(max(0.0, 1 / fps - elapsed))

        cap.release()
        frame_placeholder.empty()
        st.success("Playback complete.")

    elif not uploaded:
        st.info("Upload a video file above to get started.")


# ── Tab 2: Live Camera ────────────────────────────────────────────────────────
with tab_live:
    st.caption(
        "Uses your MacBook camera via WebRTC. Predictions are overlaid directly on the "
        "video stream in real time."
    )
    st.info("Your browser will ask for **camera permission** the first time. Click **START** to begin.", icon="ℹ️")

    ctx = webrtc_streamer(
        key="traffic-live",
        video_processor_factory=TrafficVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        st.markdown("---")
        st.subheader("Current Reading")
        if st.button("Refresh reading"):
            r = ctx.video_processor.result
            if r["label"] != "—":
                label = r["label"]
                conf  = r["confidence"]
                color = {
                    "heavy": "#c0392b", "incident": "#d35400",
                    "light": "#27ae60", "moderate": "#f39c12",
                }[label]
                st.markdown(
                    f"<div style='display:inline-block;background:{color};color:white;"
                    f"padding:6px 18px;border-radius:6px;font-size:1.4rem;font-weight:bold'>"
                    f"{label.upper()}</div>&nbsp;&nbsp;"
                    f"<span style='font-size:1.2rem'>{conf*100:.1f}% confidence</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("Waiting for first prediction…")
