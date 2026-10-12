import os
import io
import gc
import math
import zipfile
import ctypes
import joblib
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Blueberry Yield Analyzer",
    page_icon="🫐",
    layout="wide"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    1: "Sweetheart",
    2: "Elliot",
    3: "13-315",
    4: "14-321",
    5: "Cielo",
    6: "14-336",
    7: "Duke",
    8: "Star",
    9: "13-291",
    10: "Meadowlark",
    11: "14-300",
    12: "Springhigh",
    13: "14-324",
    14: "14-296",
    15: "Farthing",
    16: "Bluecrop",
    17: "Jewel",
    18: "Legacy"
}


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "models"

DEPTH_REPO_ID = "puranjit13/depthanythingv2_vkitti"
DEPTH_FILENAME = "depth_anything_v2_metric_vkitti_vits.pth"

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
GENOTYPE_MODEL_PATH = os.path.join(
    MODEL_DIR, "DINOv3_vitb16_genotype_classifier.pkl"
)
YIELD_MODEL_PATH = os.path.join(MODEL_DIR, "DINOv3_yield_ElasticNet.pkl")

DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


# ============================================================
# DEVICE / CPU CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

CPU_THREADS = int(os.getenv("TORCH_NUM_THREADS", "2"))
torch.set_num_threads(max(1, CPU_THREADS))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Streamlit reruns the script in the same process.

DEPTH_TILE_SIZE = 512
DEPTH_OVERLAP = 64

JPEG_QUALITY = 95


# ============================================================
# MEMORY MANAGEMENT
# ============================================================

def release_memory():
    """Aggressively free Python, Streamlit and C-heap memory."""
    gc.collect()
    gc.collect()

    # Return freed heap pages to the OS (Linux / Streamlit Cloud).
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_all_cache():
    """Clear stored results, Streamlit caches and reset the uploader."""
    # Stored analysis results (images, zip, metrics)
    for key in ("results", "zip_bytes"):
        st.session_state.pop(key, None)

    # Streamlit caches
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    # Changing the uploader key drops the uploaded file from memory
    st.session_state["uploader_key"] = (
        st.session_state.get("uploader_key", 0) + 1
    )

    release_memory()


if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f4f8f4;
        border: 1px solid #d8e5d8;
        text-align: center;
    }
    .yield-value {
        font-size: 38px;
        font-weight: 700;
    }
    .metric-title {
        font-size: 16px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🫐 Blueberry Yield Analyzer</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'AI-based blueberry genotype, berry detection, and yield estimation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADERS (not cached on purpose, released after use)
# ============================================================

def load_depth_model():
    """Load Depth Anything V2 with reduced peak CPU RAM."""
    config = {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    }

    depth_checkpoint = hf_hub_download(
        repo_id=DEPTH_REPO_ID,
        filename=DEPTH_FILENAME
    )

    model = DepthAnythingV2(**config, max_depth=100)

    try:
        state_dict = torch.load(
            depth_checkpoint, map_location="cpu", weights_only=True
        )
    except TypeError:
        state_dict = torch.load(depth_checkpoint, map_location="cpu")

    try:
        model.load_state_dict(state_dict, assign=True)
    except TypeError:
        model.load_state_dict(state_dict)

    del state_dict
    gc.collect()

    model.eval()
    return model


def load_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}. "
            "Upload best.pt inside the models/ directory."
        )
    return YOLO(YOLO_MODEL_PATH)


def load_dino_model():
    """Load DINOv3 using Hugging Face low-memory loading on CPU."""
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = None
    hf_token = hf_token or os.getenv("HF_TOKEN", None)

    processor_kwargs = {}
    model_kwargs = {"low_cpu_mem_usage": True}

    if hf_token:
        processor_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    processor = AutoImageProcessor.from_pretrained(
        DINO_MODEL_NAME, **processor_kwargs
    )

    try:
        model = AutoModel.from_pretrained(DINO_MODEL_NAME, **model_kwargs)
    except (TypeError, ImportError):
        model_kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModel.from_pretrained(DINO_MODEL_NAME, **model_kwargs)

    model.eval()
    return processor, model


def load_genotype_model():
    if not os.path.exists(GENOTYPE_MODEL_PATH):
        raise FileNotFoundError(
            f"Genotype model not found: {GENOTYPE_MODEL_PATH}"
        )
    return joblib.load(GENOTYPE_MODEL_PATH)


def load_yield_model():
    if not os.path.exists(YIELD_MODEL_PATH):
        raise FileNotFoundError(
            f"Yield model not found: {YIELD_MODEL_PATH}"
        )
    return joblib.load(YIELD_MODEL_PATH)


# ============================================================
# IMAGE HELPERS
# ============================================================

def bgr_to_jpeg_bytes(image_bgr, quality=JPEG_QUALITY):
    """Encode a BGR array to JPEG bytes (compact way to keep images)."""
    ok, buf = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise ValueError("Could not encode image.")
    return buf.tobytes()


# ============================================================
# FOREGROUND EXTRACTION
# ============================================================

def extract_foreground(
    image,
    depth_map,
    method="adaptive",
    threshold=None,
    percentile=55,
    smooth_kernel=7
):
    if method == "percentile":
        depth_threshold = np.percentile(depth_map, percentile)
        mask = depth_map <= depth_threshold

    elif method == "absolute":
        if threshold is None:
            threshold = np.mean(depth_map) * 0.5
        mask = depth_map <= threshold

    elif method == "adaptive":
        depth_normalized = (
            (depth_map - depth_map.min())
            / (depth_map.max() - depth_map.min() + 1e-8) * 255
        ).astype(np.uint8)

        _, mask = cv2.threshold(
            depth_normalized, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        mask = mask > 0
    else:
        raise ValueError("Unknown foreground method")

    if smooth_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (smooth_kernel, smooth_kernel)
        )
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = mask > 0

    # In-place masking: no extra full-size copy of the image
    image[~mask] = 0
    return image


def generate_foreground(raw_image, depth_model):
    """raw_image: BGR array. It is consumed (masked in place) and returned."""
    orig_h, orig_w = raw_image.shape[:2]

    tile_size = DEPTH_TILE_SIZE
    overlap = DEPTH_OVERLAP
    stride = tile_size - overlap

    new_h = math.ceil(max(1, orig_h - overlap) / stride) * stride + overlap
    new_w = math.ceil(max(1, orig_w - overlap) / stride) * stride + overlap

    resized_image = cv2.resize(
        raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )

    n_tiles_h = (new_h - overlap) // stride
    n_tiles_w = (new_w - overlap) // stride

    full_depth_map = np.zeros((new_h, new_w), dtype=np.float32)
    weight_map = np.zeros((new_h, new_w), dtype=np.float32)

    progress = st.progress(0, text="Generating foreground...")
    total_tiles = n_tiles_h * n_tiles_w
    tile_counter = 0

    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y_start = i * stride
            y_end = min(y_start + tile_size, new_h)
            x_start = j * stride
            x_end = min(x_start + tile_size, new_w)

            tile = resized_image[y_start:y_end, x_start:x_end]
            actual_h = y_end - y_start
            actual_w = x_end - x_start

            if actual_h < tile_size or actual_w < tile_size:
                tile_padded = np.zeros(
                    (tile_size, tile_size, 3), dtype=np.uint8
                )
                tile_padded[:actual_h, :actual_w] = tile
                with torch.inference_mode():
                    tile_depth = depth_model.infer_image(
                        tile_padded, tile_size
                    )
                tile_depth = tile_depth[:actual_h, :actual_w]
                del tile_padded
            else:
                with torch.inference_mode():
                    tile_depth = depth_model.infer_image(tile, tile_size)

            weight = np.ones((actual_h, actual_w), dtype=np.float32)

            if overlap > 0:
                if i > 0:
                    fade = np.linspace(0, 1, overlap)
                    weight[:overlap, :] *= fade[:, None]
                if i < n_tiles_h - 1:
                    fade = np.linspace(1, 0, overlap)
                    weight[-overlap:, :] *= fade[:, None]
                if j > 0:
                    fade = np.linspace(0, 1, overlap)
                    weight[:, :overlap] *= fade[None, :]
                if j < n_tiles_w - 1:
                    fade = np.linspace(1, 0, overlap)
                    weight[:, -overlap:] *= fade[None, :]

            full_depth_map[y_start:y_end, x_start:x_end] += tile_depth * weight
            weight_map[y_start:y_end, x_start:x_end] += weight

            tile_counter += 1
            if tile_counter == total_tiles or tile_counter % 2 == 0:
                progress.progress(
                    tile_counter / total_tiles,
                    text=(
                        f"Generating foreground... "
                        f"{tile_counter}/{total_tiles} tiles"
                    )
                )
            del tile_depth, weight, tile

    del resized_image

    full_depth_map /= np.maximum(weight_map, 1e-8)
    del weight_map

    full_depth_map = cv2.resize(
        full_depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )

    foreground = extract_foreground(
        raw_image,
        full_depth_map,
        method="adaptive",
        threshold=13.0,
        percentile=55,
        smooth_kernel=7
    )

    del full_depth_map
    progress.empty()
    return foreground


# ============================================================
# YOLO BERRY DETECTION + ANNOTATED IMAGE
# ============================================================

def predict_berry_count(foreground, model, tile_size=640, conf=0.4):
    """CPU-safe YOLO tiling with only one tile resident at a time."""
    original_h, original_w = foreground.shape[:2]
    output_image = foreground.copy()

    padded_w = math.ceil(original_w / tile_size) * tile_size
    padded_h = math.ceil(original_h / tile_size) * tile_size

    padded_image = cv2.copyMakeBorder(
        foreground, 0, padded_h - original_h, 0, padded_w - original_w,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    n_x = padded_w // tile_size
    n_y = padded_h // tile_size
    total_tiles = n_x * n_y
    tile_counter = 0
    total_green = 0
    total_ripe = 0

    progress = st.progress(0, text="Detecting berries...")

    for y in range(0, padded_h, tile_size):
        for x in range(0, padded_w, tile_size):
            tile = padded_image[y:y + tile_size, x:x + tile_size]

            results = model.predict(
                source=tile,
                conf=conf,
                imgsz=tile_size,
                device="cpu",
                workers=0,
                batch=1,
                verbose=False
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, class_id in zip(boxes, class_ids):
                    class_name = result.names[class_id]
                    class_name_lower = class_name.lower()

                    if class_name_lower == "green":
                        total_green += 1
                        color = (0, 255, 0)
                    elif class_name_lower == "ripe":
                        total_ripe += 1
                        color = (0, 0, 255)
                    else:
                        continue

                    x1, y1, x2, y2 = box
                    x1 = int(x1 + x)
                    y1 = int(y1 + y)
                    x2 = int(x2 + x)
                    y2 = int(y2 + y)

                    if (x1 >= original_w or y1 >= original_h
                            or x2 <= 0 or y2 <= 0):
                        continue

                    x1 = max(0, min(x1, original_w - 1))
                    y1 = max(0, min(y1, original_h - 1))
                    x2 = max(0, min(x2, original_w - 1))
                    y2 = max(0, min(y2, original_h - 1))

                    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        output_image, class_name, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

            del results, result, tile
            tile_counter += 1

            if tile_counter == total_tiles or tile_counter % 2 == 0:
                progress.progress(
                    tile_counter / total_tiles,
                    text=(
                        f"Detecting berries... "
                        f"{tile_counter}/{total_tiles} tiles"
                    )
                )

    progress.empty()
    del padded_image

    total_berries = total_green + total_ripe
    return total_green, total_ripe, total_berries, output_image


# ============================================================
# DINOv3 GENOTYPE PREDICTION
# ============================================================

def predict_genotype(foreground, processor, dino_model, genotype_model):
    rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    num_register_tokens = getattr(
        dino_model.config, "num_register_tokens", 4
    )

    with torch.inference_mode():
        outputs = dino_model(**inputs)
        hidden = outputs.last_hidden_state
        patch_embeddings = hidden[:, 1 + num_register_tokens:, :]
        embedding = patch_embeddings.mean(dim=1).cpu().numpy()

    del rgb, image, inputs, outputs, hidden, patch_embeddings

    genotype_id = int(genotype_model.predict(embedding)[0])
    genotype_name = CLASS_NAMES.get(
        genotype_id, f"Unknown class ({genotype_id})"
    )
    return genotype_id, genotype_name


# ============================================================
# ZIP BUILDER
# ============================================================

def build_zip(original_bytes, original_ext, foreground_jpg,
              detection_jpg, summary_df):
    """Bundle the three images (+ summary CSV) into an in-memory zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # JPEGs are already compressed, so store them without re-deflating
        zf.writestr(
            f"1_original_image{original_ext}", original_bytes,
            compress_type=zipfile.ZIP_STORED
        )
        zf.writestr(
            "2_foreground_image.jpg", foreground_jpg,
            compress_type=zipfile.ZIP_STORED
        )
        zf.writestr(
            "3_berry_detection_result.jpg", detection_jpg,
            compress_type=zipfile.ZIP_STORED
        )
        zf.writestr(
            "4_prediction_summary.csv",
            summary_df.to_csv(index=False).encode("utf-8")
        )
    return buf.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.write(
        "**Please upload an image for a Blueberry plant and an "
        "approximate average berry wt. to get started**"
    )
    st.write(
        "The Blueberry Yield Analyzer tool performs the following "
        "to estimate Plant Yield"
    )
    st.divider()
    st.write("**Step 1: Foreground Extraction:**")
    st.write("Model used: Depth Anything V2 ViT-S")

    st.write("**Step 2: Blueberry detection based on Maturity:**")
    st.write("YOLO11l")

    st.write("**Step 3: Yield Potential estimation using DINOv3:**")
    st.write("DINOv3 ViT-B/16")

    st.write("**Step 4: Yield Estimation (in g):**")
    st.write("ElasticNet Regressor")

    st.divider()
    st.write(f"Compute device: `{DEVICE}`")
    st.caption("CPU-optimized configuration for Streamlit Cloud")

    st.divider()
    if st.button(
        "🧹 Clear cache & memory",
        use_container_width=True,
        help="Removes stored results, uploaded image and all cached "
             "memory. Download the zip first!"
    ):
        clear_all_cache()
        st.rerun()


# ============================================================
# RESULTS RENDERING (reads only from session_state)
# ============================================================

def render_results(res):
    st.subheader("1️⃣ Original Image")
    st.image(res["original_bytes"], use_container_width=True)
    st.divider()

    st.subheader("2️⃣ Foreground Extraction")
    st.image(
        res["foreground_jpg"],
        use_container_width=True,
        caption="Foreground extracted from Original Image"
    )

    st.subheader("3️⃣ Berry Detection")
    st.image(
        res["detection_jpg"],
        use_container_width=True,
        caption=(
            "[**Detected berries** "
            "(**Green** = green berries, **Red** = ripe berries)]"
        )
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Green Berries", res["green"])
    col2.metric("🔴 Ripe Berries", res["ripe"])
    col3.metric("🫐 Total Berries", res["total"])

    st.subheader("4️⃣ Genotype Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Genotype", res["genotype"])
    col2.metric("Yield Potential Estimation", res["yield_text"])

    st.subheader("5️⃣ Yield Prediction")
    st.success("Analysis completed successfully!")
    st.divider()

    st.markdown(
        """
        <div class="result-box">
            <div class="metric-title">ESTIMATED BLUEBERRY YIELD</div>
            <div class="yield-value">{:.2f} g</div>
        </div>
        """.format(res["predicted_yield"]),
        unsafe_allow_html=True
    )
    st.divider()

    st.subheader("📋 Prediction Summary")
    st.dataframe(
        res["summary"], hide_index=True, use_container_width=True
    )

    st.divider()
    st.subheader("⬇️ Download Results")
    st.download_button(
        label="📦 Download results (.zip)",
        data=res["zip_bytes"],
        file_name=f"{res['base_name']}_blueberry_results.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    st.caption(
        "Zip contains: original image, foreground image, berry detection "
        "result and the prediction summary (CSV). When you are done, use "
        "**Clear cache & memory** in the sidebar."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload Blueberry Image")

uploaded_file = st.file_uploader(
    "Choose an original blueberry plant image",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state['uploader_key']}"
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # Show the preview only before an analysis exists
    # (afterwards the original image is shown inside the results).
    if "results" not in st.session_state:
        st.subheader("1️⃣ Original Image")
        st.image(uploaded_file, use_container_width=True)
        st.divider()

    berry_weight = st.number_input(
        "Average berry weight (g)",
        min_value=0.01,
        max_value=100.0,
        value=2.0,
        step=0.01
    )

    run_button = st.button(
        "🫐 Analyze Image",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # RUN COMPLETE PIPELINE
    # ========================================================
    if run_button:

        # Drop any previous result before starting a new run
        st.session_state.pop("results", None)
        release_memory()

        depth_model = None
        yolo_model = None
        processor = None
        dino_model = None
        genotype_model = None
        yield_model = None
        raw_image = None
        foreground = None
        annotated_image = None

        try:
            original_bytes = uploaded_file.getvalue()
            base_name, original_ext = os.path.splitext(uploaded_file.name)
            original_ext = original_ext.lower() or ".jpg"

            # Decode directly from memory (no temp file on disk)
            raw_image = cv2.imdecode(
                np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if raw_image is None:
                raise ValueError("Could not read uploaded image.")

            # -------------------------------------------------
            # 1. FOREGROUND GENERATION
            # -------------------------------------------------
            with st.spinner(
                "Loading Depth Anything V2 on CPU and "
                "generating foreground..."
            ):
                depth_model = load_depth_model()
                foreground = generate_foreground(raw_image, depth_model)
                raw_image = None  # masked in place -> same object as foreground

            del depth_model
            depth_model = None
            release_memory()

            # Store compact JPEG bytes instead of big arrays
            foreground_jpg = bgr_to_jpeg_bytes(foreground)

            # -------------------------------------------------
            # 2. BERRY DETECTION
            # -------------------------------------------------
            with st.spinner(
                "Loading YOLO and detecting green and ripe "
                "berries on CPU..."
            ):
                yolo_model = load_yolo_model()
                green, ripe, total, annotated_image = predict_berry_count(
                    foreground, yolo_model, tile_size=640, conf=0.4
                )

            del yolo_model
            yolo_model = None
            release_memory()

            detection_jpg = bgr_to_jpeg_bytes(annotated_image)
            del annotated_image
            annotated_image = None

            # -------------------------------------------------
            # 3. GENOTYPE PREDICTION
            # -------------------------------------------------
            with st.spinner(
                "Loading DINOv3 on CPU and predicting genotype..."
            ):
                processor, dino_model = load_dino_model()
                genotype_model = load_genotype_model()

                genotype_id, genotype = predict_genotype(
                    foreground, processor, dino_model, genotype_model
                )

            del processor, dino_model, genotype_model, foreground
            processor = dino_model = genotype_model = foreground = None
            release_memory()

            if genotype_id <= 7:
                yield_text = "Low yield potential"
            elif genotype_id <= 12:
                yield_text = "Average yield potential"
            else:
                yield_text = "High yield potential"

            # -------------------------------------------------
            # 4. YIELD PREDICTION
            # -------------------------------------------------
            with st.spinner("Estimating blueberry yield..."):
                yield_model = load_yield_model()

                X_new = pd.DataFrame({
                    "Genotype": [genotype_id],
                    "Avg. Berry wt": [berry_weight],
                    "DL berry count": [ripe]
                })

                predicted_yield = float(yield_model.predict(X_new)[0])

            del yield_model, X_new
            yield_model = None
            release_memory()

            # -------------------------------------------------
            # SUMMARY + ZIP
            # -------------------------------------------------
            summary = pd.DataFrame({
                "Parameter": [
                    "Predicted Genotype",
                    "Average Berry Weight",
                    "DL Ripe Berry Count",
                    "Green Berry Count",
                    "Total Berry Count",
                    "Yield Potential Estimation",
                    "Predicted Yield"
                ],
                "Value": [
                    genotype,
                    f"{berry_weight:.2f} g",
                    str(ripe),
                    str(green),
                    str(total),
                    yield_text,
                    f"{predicted_yield:.2f} g"
                ]
            })

            zip_bytes = build_zip(
                original_bytes, original_ext,
                foreground_jpg, detection_jpg, summary
            )

            st.session_state["results"] = {
                "original_bytes": original_bytes,
                "foreground_jpg": foreground_jpg,
                "detection_jpg": detection_jpg,
                "green": green,
                "ripe": ripe,
                "total": total,
                "genotype": genotype,
                "yield_text": yield_text,
                "predicted_yield": predicted_yield,
                "summary": summary,
                "zip_bytes": zip_bytes,
                "base_name": base_name or "image",
            }

        except Exception as e:
            st.error("An error occurred during analysis.")
            st.exception(e)

        finally:
            # Always release model memory
            depth_model = yolo_model = processor = dino_model = None
            genotype_model = yield_model = None
            raw_image = foreground = annotated_image = None
            release_memory()

    # Render stored results (survives reruns such as the download click)
    if "results" in st.session_state:
        render_results(st.session_state["results"])

else:
    # No file selected (first visit, or after clearing)
    if "results" in st.session_state:
        render_results(st.session_state["results"])
    else:
        st.info(
            "Upload an original blueberry plant image "
            "to begin the analysis."
        )
