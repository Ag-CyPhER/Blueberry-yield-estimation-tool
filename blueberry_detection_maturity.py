import os
import io
import math
import base64
import zipfile
import cv2
import torch
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(
    page_title="Blueberry Detection and Maturity Assessment",
    page_icon="🫐",
    layout="wide"
)

# Detection label ids used internally
LABEL_IDS = {"green": 0, "ripe": 1}
LABEL_NAMES = {0: "green", 1: "ripe"}
LABEL_COLORS_BGR = {0: (0, 255, 0), 1: (211, 0, 148)}

# ============================================================
# MODEL PATHS
# ============================================================
MODEL_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

# Optional faster CPU model. Create it once with:
#   from ultralytics import YOLO
#   YOLO("models/best.pt").export(format="openvino", imgsz=640)
# If this folder exists it is used automatically.
YOLO_OPENVINO_DIR = os.path.join(MODEL_DIR, "best_openvino_model")

# Longest image side (px) used for the on-screen viewer.
# Display only - the analysis and the downloaded files use full resolution.
DISPLAY_MAX_SIDE = 1800

# YOLO is run ONCE at this low confidence; the sidebar slider then simply
# filters the stored detections (instant, no re-inference).
CONF_FLOOR = 0.05

YOLO_TILE = 640
YOLO_BATCH = 6

# The image is resized to this fraction of its size (same factor for width
# and height, so the aspect ratio never changes) BEFORE detection.
# Set to 1.0 to analyse at full resolution.
RESIZE_SCALE = 0.5

# ============================================================
# DEVICE (CPU only)
# ============================================================
DEVICE = torch.device("cpu")

torch.set_num_threads(max(1, os.cpu_count() or 1))

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>

    .main-title {
        font-size: clamp(26px, 5vw, 42px);
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: clamp(14px, 2.5vw, 18px);
        color: #666;
        margin-bottom: 30px;
    }

    /* Always reserve scrollbar space so the layout never jumps sideways */
    [data-testid="stMain"], section.main {
        overflow-y: scroll !important;
        scrollbar-gutter: stable;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">🫐 Blueberry Detection and Maturity Assessment</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'AI-based blueberry detection and maturity assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_yolo_model():
    """Returns (model, is_openvino)."""

    if os.path.isdir(YOLO_OPENVINO_DIR):
        return YOLO(YOLO_OPENVINO_DIR, task="detect"), True

    return YOLO(YOLO_MODEL_PATH), False


# ============================================================
# YOLO DETECTION (runs once per image, at a low confidence floor)
# ============================================================
def detect_raw(image_bgr, model, is_openvino, tile_size=YOLO_TILE):
    """
    Tiles the image, runs YOLO on CPU once at CONF_FLOOR and
    returns every detection in full-image coordinates:

        {"boxes": (N,4) float32, "scores": (N,), "labels": (N,) int}

    labels: 0 = green (immature), 1 = ripe.
    The confidence slider filters these arrays afterwards, so moving the
    slider never needs another model run.
    """

    h, w = image_bgr.shape[:2]

    padded_w = math.ceil(w / tile_size) * tile_size
    padded_h = math.ceil(h / tile_size) * tile_size

    padded = cv2.copyMakeBorder(
        image_bgr,
        0, padded_h - h,
        0, padded_w - w,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    tiles = []
    positions = []

    for y in range(0, padded_h, tile_size):
        for x in range(0, padded_w, tile_size):
            tiles.append(padded[y:y + tile_size, x:x + tile_size])
            positions.append((x, y))

    all_boxes, all_scores, all_labels = [], [], []

    predict_kwargs = dict(
        conf=CONF_FLOOR,
        imgsz=tile_size,
        verbose=False,
    )

    if not is_openvino:
        predict_kwargs.update(device="cpu", batch=YOLO_BATCH, half=False)

    progress = st.progress(0, text="Detecting blueberries...")

    for b in range(0, len(tiles), YOLO_BATCH):

        chunk = tiles[b:b + YOLO_BATCH]
        chunk_pos = positions[b:b + YOLO_BATCH]

        with torch.inference_mode():
            results = model.predict(source=chunk, **predict_kwargs)

        for result, (tx, ty) in zip(results, chunk_pos):

            if result.boxes is None or len(result.boxes) == 0:
                continue

            lab_map = {
                int(i): LABEL_IDS.get(str(n).lower(), -1)
                for i, n in result.names.items()
            }

            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf_ = result.boxes.conf.cpu().numpy().astype(np.float32)
            cls = result.boxes.cls.cpu().numpy().astype(int)

            labels = np.array([lab_map.get(int(c), -1) for c in cls])

            xyxy += np.array([tx, ty, tx, ty], dtype=np.float32)

            # drop unknown classes and boxes lying fully in the padding
            keep = (labels >= 0) & (xyxy[:, 0] < w) & (xyxy[:, 1] < h)

            if not keep.any():
                continue

            xyxy = xyxy[keep]
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)

            all_boxes.append(xyxy)
            all_scores.append(conf_[keep])
            all_labels.append(labels[keep])

        progress.progress(
            min(1.0, (b + YOLO_BATCH) / len(tiles)),
            text=f"Detecting blueberries... "
                 f"{min(b + YOLO_BATCH, len(tiles))}/{len(tiles)} tiles"
        )

    progress.empty()

    if all_boxes:
        return {
            "boxes": np.concatenate(all_boxes),
            "scores": np.concatenate(all_scores),
            "labels": np.concatenate(all_labels),
        }

    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=int),
    }


def filter_dets(raw, conf):
    """Keep detections with score >= conf (instant)."""

    keep = raw["scores"] >= (conf - 1e-9)

    return raw["boxes"][keep], raw["labels"][keep]


def draw_detections(img_bgr, boxes, labels, factor=1.0):
    """
    Draws boxes only (no text labels - the viewer shows a colour legend)
    on a copy of img_bgr. `factor` rescales box coordinates (used when
    drawing on the smaller display image). Line width scales with the
    image so boxes stay visible on big photos.
    """

    out = img_bgr.copy()

    h, w = out.shape[:2]

    s = max(1.0, max(h, w) / 1500.0)
    thickness = max(1, int(round(2 * s)))

    for (x1, y1, x2, y2), lab in zip(boxes * factor, labels):

        color = LABEL_COLORS_BGR[int(lab)]

        cv2.rectangle(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )

    return out


# ============================================================
# DISPLAY HELPERS
# ============================================================
def make_display(image_bgr, max_side=DISPLAY_MAX_SIDE):
    """Downscaled copy for on-screen use. Returns (image, factor)."""

    h, w = image_bgr.shape[:2]

    f = min(1.0, max_side / max(h, w))

    if f >= 1.0:
        return image_bgr, 1.0

    disp = cv2.resize(
        image_bgr,
        (int(w * f), int(h * f)),
        interpolation=cv2.INTER_AREA
    )

    return disp, f


def to_b64_jpeg(img_bgr, quality=85):

    ok, buf = cv2.imencode(
        ".jpg",
        img_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_zip(detection, boxes, labels, conf):
    """ZIP containing the analysed image and its detection image, same size."""

    annotated = draw_detections(
        detection["image_bgr"],
        boxes,
        labels,
        1.0
    )

    ok, buf = cv2.imencode(
        ".jpg",
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    ok_o, buf_o = cv2.imencode(
        ".jpg",
        detection["image_bgr"],
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    stem, ext = os.path.splitext(detection["name"])

    zbuf = io.BytesIO()

    # ZIP_STORED: JPEG/PNG data is already compressed, so this is fastest
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_STORED) as zf:

        zf.writestr(f"{stem}_original.jpg", buf_o.tobytes())
        zf.writestr(f"{stem}_detected_conf{conf:.2f}.jpg", buf.tobytes())

    return zbuf.getvalue()


def pil_to_bgr(pil_img):

    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def resize_for_analysis(image_bgr):
    """Apply RESIZE_SCALE (aspect ratio preserved)."""

    if RESIZE_SCALE == 1.0:
        return image_bgr

    h, w = image_bgr.shape[:2]

    new_w = max(1, int(round(w * RESIZE_SCALE)))
    new_h = max(1, int(round(h * RESIZE_SCALE)))

    return cv2.resize(
        image_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# SLIDER-ZOOM SIDE-BY-SIDE VIEWER (HTML + JS)
# The zoom slider lives INSIDE the viewer (below the images), so moving it
# is instant and never reruns the Streamlit app.
# ============================================================
VIEWER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent;
               font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
               color: #8a8a8a; }
  .row   { display: flex; gap: 14px; }
  .panel { flex: 1 1 0; min-width: 0; text-align: center; }
  .title { font-weight: 700; font-size: 18px; margin: 2px 0 8px; }
  .clip  { overflow: hidden; margin: 0 auto; line-height: 0;
           box-shadow: 0 0 0 1px #c8c8c8; border-radius: 4px;
           touch-action: none; background: #000; cursor: default; }
  .pan   { transform-origin: 0 0; will-change: transform; }
  .pan img { display: block; width: 100%; height: 100%;
             user-select: none; -webkit-user-drag: none; }
  .legend { margin-top: 10px; text-align: center; font-size: 14px; }
  .legend span.sw { display: inline-block; width: 14px; height: 14px;
                    border: 3px solid; box-sizing: border-box;
                    vertical-align: -2px; margin: 0 6px 0 16px; }
  .zoombar { margin: 16px auto 0; max-width: 620px; text-align: center;
             font-size: 15px; }
  .zoombar input[type=range] { width: 100%; margin: 8px 0 2px; }
  .zoombar .val { font-weight: 700; }
</style>
</head>
<body>
  <div class="row">
    <div class="panel">
      <div class="title">1. Original Image</div>
      <div class="clip"><div class="pan"><img src="data:image/jpeg;base64,__ORIG__"></div></div>
    </div>
    <div class="panel">
      <div class="title">2. Detected Blueberries (confidence &ge; __CONF__)</div>
      <div class="clip"><div class="pan"><img src="data:image/jpeg;base64,__DET__"></div></div>
    </div>
  </div>
  <div class="legend">
    <span class="sw" style="border-color:#00e000"></span>Immature blueberries
    <span class="sw" style="border-color:#9400D3"></span>Mature blueberries
  </div>

  <div class="zoombar">
    <div>Adjust slide bar to zoom in and out in an image</div>
    <input id="zoom" type="range" min="1" max="10" step="0.05" value="1">
    <div>Zoom: <span class="val" id="zval">1.0x</span>
      &nbsp;(drag an image to move around when zoomed in)</div>
  </div>

<script>
(function () {
  const H = __H__, MAXZ = 10;

  const panels = Array.from(document.querySelectorAll('.panel')).map(el => ({
    col: el,
    clip: el.querySelector('.clip'),
    pan: el.querySelector('.pan'),
    img: el.querySelector('img'),
    cw: 1, ch: 1
  }));

  // shared view state (both images always zoom/pan together).
  // tx, ty are fractions of the displayed image width/height.
  let s = 1, tx = 0, ty = 0;

  const slider = document.getElementById('zoom');
  const zval = document.getElementById('zval');

  function layout() {
    panels.forEach(p => {
      const nw = p.img.naturalWidth, nh = p.img.naturalHeight;
      if (!nw) return;
      const colW = p.col.clientWidth;
      const fit = Math.min(colW / nw, H / nh);
      p.cw = Math.floor(nw * fit);
      p.ch = Math.floor(nh * fit);
      p.clip.style.width = p.cw + 'px';
      p.clip.style.height = p.ch + 'px';
    });
    apply();
  }

  function clampView() {
    tx = Math.min(0, Math.max(1 - s, tx));
    ty = Math.min(0, Math.max(1 - s, ty));
  }

  function apply() {
    clampView();
    panels.forEach(p => {
      p.pan.style.transform =
        'translate(' + (tx * p.cw) + 'px,' + (ty * p.ch) + 'px) scale(' + s + ')';
      p.clip.style.cursor = s > 1 ? 'grab' : 'default';
    });
    zval.textContent = s.toFixed(1) + 'x';
  }

  // slider: zoom around the centre of the current view
  slider.addEventListener('input', () => {
    const ns = Math.min(MAXZ, Math.max(1, parseFloat(slider.value)));
    const cx = (0.5 - tx) / s, cy = (0.5 - ty) / s;
    tx = 0.5 - cx * ns;
    ty = 0.5 - cy * ns;
    s = ns;
    apply();
  });

  // drag to pan (mouse / touch / pen)
  let drag = null;

  panels.forEach(p => {
    p.clip.addEventListener('pointerdown', e => {
      if (s <= 1) return;
      drag = { x: e.clientX, y: e.clientY, p: p };
      p.clip.setPointerCapture(e.pointerId);
      p.clip.style.cursor = 'grabbing';
      e.preventDefault();
    });

    p.clip.addEventListener('pointermove', e => {
      if (!drag || drag.p !== p) return;
      tx += (e.clientX - drag.x) / p.cw;
      ty += (e.clientY - drag.y) / p.ch;
      drag.x = e.clientX; drag.y = e.clientY;
      apply();
    });

    const end = () => { drag = null; apply(); };
    p.clip.addEventListener('pointerup', end);
    p.clip.addEventListener('pointercancel', end);

    // double click resets the view
    p.clip.addEventListener('dblclick', () => {
      s = 1; tx = 0; ty = 0; slider.value = 1; apply();
    });

    if (p.img.complete) layout();
    else p.img.addEventListener('load', layout);
  });

  window.addEventListener('resize', layout);
  layout();
})();
</script>
</body>
</html>
"""


def render_viewer(orig_b64, det_b64, disp_h, disp_w, conf):

    viewer_h = int(min(720, max(380, 620 * disp_h / disp_w)))

    html = (
        VIEWER_TEMPLATE
        .replace("__ORIG__", orig_b64)
        .replace("__DET__", det_b64)
        .replace("__CONF__", f"{conf:.2f}")
        .replace("__H__", str(viewer_h))
    )

    components.html(html, height=viewer_h + 190, scrolling=False)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.divider()

    st.write("**Blueberry detection based on maturity:**")
    st.write("YOLO11l")

    st.divider()

    st.write(f"Inference running on `{DEVICE}`")

    st.subheader("🎚️ Detection settings")

    conf = st.slider(
        "Detection confidence threshold",
        min_value=CONF_FLOOR,
        max_value=0.95,
        value=0.40,
        step=0.05,
        help=(
            "Only blueberries detected with at least this confidence are "
            "shown and counted. Lower = more detections (more false "
            "positives), higher = fewer but surer detections. Changing it "
            "is instant - the model is not re-run."
        )
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================
st.header("📷 Upload Blueberry Image")

st.write(
    "Upload an image of a blueberry plant to detect immature and mature "
    "blueberries."
)

uploaded_file = st.file_uploader(
    "Choose an original blueberry plant image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# MAIN APPLICATION
# ============================================================
if uploaded_file is None:

    for _k in ("upload_id", "src_pil", "detection", "zip", "analyzed_id"):
        st.session_state.pop(_k, None)

    st.info(
        "Upload an original blueberry plant image "
        "to begin the analysis."
    )

else:

    upload_id = (
        getattr(uploaded_file, "file_id", None)
        or f"{uploaded_file.name}-{uploaded_file.size}"
    )

    # New upload -> reset everything and decode the image once
    if st.session_state.get("upload_id") != upload_id:

        st.session_state["upload_id"] = upload_id

        for _k in ("detection", "zip"):
            st.session_state.pop(_k, None)

        try:
            pil = Image.open(io.BytesIO(uploaded_file.getvalue()))
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            st.session_state["src_pil"] = pil
        except Exception as e:
            st.session_state.pop("src_pil", None)
            st.error("Could not read uploaded image.")
            st.exception(e)

    src_pil = st.session_state.get("src_pil")

    # --------------------------------------------------------
    # STAGE 1 (runs ONCE per uploaded image, automatically):
    # YOLO detection at the confidence floor -> stored raw detections
    # --------------------------------------------------------

    if src_pil is not None and st.session_state.get("analyzed_id") != upload_id:

        st.session_state["analyzed_id"] = upload_id

        st.session_state["analysis_id"] = (
            st.session_state.get("analysis_id", 0) + 1
        )
        st.session_state.pop("detection", None)
        st.session_state.pop("zip", None)

        try:

            with st.spinner("Loading AI models..."):

                yolo_model, is_ov = load_yolo_model()

            original_bgr = resize_for_analysis(pil_to_bgr(src_pil))

            with st.spinner(
                "Detecting immature and mature blueberries "
            ):

                raw = detect_raw(original_bgr, yolo_model, is_ov)

            with st.spinner("Preparing viewer..."):

                disp_bgr, disp_factor = make_display(original_bgr)

                orig_b64 = to_b64_jpeg(disp_bgr)

            st.session_state["detection"] = {
                "name": uploaded_file.name,
                "image_bgr": original_bgr,   # image used everywhere
                "disp_bgr": disp_bgr,
                "disp_factor": disp_factor,
                "orig_b64": orig_b64,
                "raw": raw,
            }

        except Exception as e:

            st.error("An error occurred during image analysis.")

            st.exception(e)

    detection = st.session_state.get("detection")

    if detection is not None:

        st.divider()

        # ----------------------------------------------------
        # Apply the confidence slider (instant filter + redraw)
        # ----------------------------------------------------

        boxes, labels = filter_dets(detection["raw"], conf)

        n_green = int((labels == 0).sum())
        n_ripe = int((labels == 1).sum())
        n_total = n_green + n_ripe

        disp_annotated = draw_detections(
            detection["disp_bgr"],
            boxes,
            labels,
            detection["disp_factor"]
        )

        det_b64 = to_b64_jpeg(disp_annotated)

        # ----------------------------------------------------
        # Visualization: Original | Detections (side by side)
        # ----------------------------------------------------

        st.subheader("🔍 Image Analysis")

        st.caption(
            "Adjust the confidence threshold for the blueberry detection "
            "model in the sidebar. Use the slide bar below the images to "
            "zoom in and out."
        )

        dh, dw = detection["disp_bgr"].shape[:2]

        render_viewer(
            detection["orig_b64"],
            det_b64,
            dh, dw,
            conf
        )

        # ----------------------------------------------------
        # Blueberry detection counts
        # ----------------------------------------------------

        st.subheader("🫐 Blueberry Detection Count")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🟢 Immature Blueberries", n_green)

        with col2:
            st.metric("🟣 Mature Blueberries", n_ripe)

        with col3:
            st.metric("🫐 Total Blueberries", n_total)

        # ----------------------------------------------------
        # Download ZIP (analysed image + detection image)
        # ----------------------------------------------------

        zip_key = (
            st.session_state.get("analysis_id", 0),
            round(float(conf), 2)
        )

        cached_zip = st.session_state.get("zip")

        if cached_zip is None or cached_zip[0] != zip_key:

            with st.spinner("Preparing download..."):

                zip_bytes = build_zip(detection, boxes, labels, conf)

            st.session_state["zip"] = (zip_key, zip_bytes)

        else:

            zip_bytes = cached_zip[1]

        stem = os.path.splitext(detection["name"])[0]

        st.download_button(
            "⬇️ Download original + detection images (ZIP)",
            data=zip_bytes,
            file_name=f"{stem}_blueberry_detection.zip",
            mime="application/zip"
        )
