import os
import math
import tempfile
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gc
import contextlib
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO


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

from huggingface_hub import hf_hub_download

DEPTH_REPO_ID = "puranjit13/depthanythingv2_vkitti"
DEPTH_FILENAME = "depth_anything_v2_metric_vkitti_vits.pth"

YOLO_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best.pt"
)

GENOTYPE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "DINOv3_vitb16_genotype_classifier.pkl"
)

YIELD_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "DINOv3_yield_ElasticNet.pkl"
)

DINO_MODEL_NAME = (
    "facebook/dinov3-vitb16-pretrain-lvd1689m"
    # DEPTH_REPO_ID,
    # "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

# ============================================================
# DEVICE
# ============================================================

# ============================================================
# CPU CONFIGURATION FOR STREAMLIT CLOUD
# ============================================================
# Streamlit Cloud normally runs without an NVIDIA GPU.
# Force CPU inference so the app behaves consistently in the cloud.
DEVICE = torch.device("cpu")

# Limit PyTorch CPU threads so one Streamlit instance does not
# consume all available CPU resources.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

CPU_THREADS = int(os.getenv("TORCH_NUM_THREADS", "2"))
torch.set_num_threads(max(1, CPU_THREADS))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Streamlit may rerun this script in the same Python process.
    pass

# Keep the large-model workflow sequential to reduce RAM usage.
CPU_ONLY = True

# Depth inference is the most expensive part of this application.
# Downscale only the depth-estimation input when an uploaded image
# is extremely large. The final foreground is returned at the
# original image resolution.
DEPTH_MAX_SIDE = 1600
DEPTH_TILE_SIZE = 512
DEPTH_OVERLAP = 64


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


# ============================================================
# HEADER
# ============================================================

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
# LOAD DEPTH ANYTHING V2
# ============================================================

def load_depth_model():
    """Load Depth Anything V2 with reduced peak CPU RAM."""
    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384]
        }
    }

    depth_checkpoint = hf_hub_download(
        repo_id=DEPTH_REPO_ID,
        filename=DEPTH_FILENAME
    )

    model = DepthAnythingV2(
        **model_configs["vits"],
        max_depth=100
    )

    # weights_only avoids unnecessary pickle overhead. assign=True avoids
    # an extra parameter copy on supported PyTorch versions.
    try:
        state_dict = torch.load(
            depth_checkpoint,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(
            depth_checkpoint,
            map_location="cpu"
        )

    try:
        model.load_state_dict(state_dict, assign=True)
    except TypeError:
        model.load_state_dict(state_dict)

    del state_dict
    gc.collect()

    model.eval()
    return model


# ============================================================
# LOAD YOLO
# ============================================================

def load_yolo_model():
    """Load YOLO without caching so it can be released after detection."""
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}. "
            "Upload best.pt inside the models/ directory."
        )

    return YOLO(YOLO_MODEL_PATH)


# ============================================================
# LOAD DINOv3
# ============================================================

def load_dino_model():
    """Load DINOv3 using Hugging Face low-memory loading on CPU."""
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = None

    hf_token = hf_token or os.getenv("HF_TOKEN", None)

    processor_kwargs = {}
    model_kwargs = {
        "low_cpu_mem_usage": True
    }

    if hf_token:
        processor_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    processor = AutoImageProcessor.from_pretrained(
        DINO_MODEL_NAME,
        **processor_kwargs
    )

    try:
        model = AutoModel.from_pretrained(
            DINO_MODEL_NAME,
            **model_kwargs
        )
    except (TypeError, ImportError):
        # Fallback for older Transformers/without accelerate.
        model_kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModel.from_pretrained(
            DINO_MODEL_NAME,
            **model_kwargs
        )

    model.eval()
    return processor, model


# ============================================================
# LOAD GENOTYPE CLASSIFIER
# ============================================================

def load_genotype_model():

    if not os.path.exists(GENOTYPE_MODEL_PATH):
        raise FileNotFoundError(
            f"Genotype model not found: {GENOTYPE_MODEL_PATH}"
        )

    return joblib.load(GENOTYPE_MODEL_PATH)


# ============================================================
# LOAD YIELD MODEL
# ============================================================

def load_yield_model():

    if not os.path.exists(YIELD_MODEL_PATH):
        raise FileNotFoundError(
            f"Yield model not found: {YIELD_MODEL_PATH}"
        )

    return joblib.load(YIELD_MODEL_PATH)


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

        depth_threshold = np.percentile(
            depth_map,
            percentile
        )

        mask = depth_map <= depth_threshold

    elif method == "absolute":

        if threshold is None:
            threshold = np.mean(depth_map) * 0.5

        mask = depth_map <= threshold

    elif method == "adaptive":

        depth_normalized = (
            (depth_map - depth_map.min()) /
            (
                depth_map.max()
                - depth_map.min()
                + 1e-8
            ) * 255
        ).astype(np.uint8)

        _, mask = cv2.threshold(
            depth_normalized,
            0,
            255,
            cv2.THRESH_BINARY_INV +
            cv2.THRESH_OTSU
        )

        mask = mask > 0

    else:

        raise ValueError(
            "Unknown foreground method"
        )

    if smooth_kernel > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                smooth_kernel,
                smooth_kernel
            )
        )

        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = mask > 0

    foreground = image.copy()

    foreground[~mask] = 0

    return foreground


# ============================================================
# GENERATE FOREGROUND
# ============================================================

def generate_foreground(
    image_path,
    depth_model
):

    raw_image = cv2.imread(
        image_path
    )

    if raw_image is None:

        raise ValueError(
            "Could not read uploaded image."
        )

    orig_h, orig_w = raw_image.shape[:2]

    # Preserve the original inference geometry so predictions are not
    # intentionally changed by a cloud-only resize.
    tile_size = DEPTH_TILE_SIZE
    overlap = DEPTH_OVERLAP
    depth_h = orig_h
    depth_w = orig_w
    stride = tile_size - overlap

    new_h = (
        math.ceil(
            max(1, depth_h - overlap) / stride
        ) * stride + overlap
    )

    new_w = (
        math.ceil(
            max(1, depth_w - overlap) / stride
        ) * stride + overlap
    )

    resized_image = cv2.resize(
        raw_image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    n_tiles_h = (
        new_h - overlap
    ) // stride

    n_tiles_w = (
        new_w - overlap
    ) // stride

    full_depth_map = np.zeros(
        (new_h, new_w),
        dtype=np.float32
    )

    weight_map = np.zeros(
        (new_h, new_w),
        dtype=np.float32
    )

    progress = st.progress(
        0,
        text="Generating foreground..."
    )

    total_tiles = (
        n_tiles_h * n_tiles_w
    )

    tile_counter = 0

    for i in range(n_tiles_h):

        for j in range(n_tiles_w):

            y_start = i * stride
            y_end = min(
                y_start + tile_size,
                new_h
            )

            x_start = j * stride
            x_end = min(
                x_start + tile_size,
                new_w
            )

            tile = resized_image[
                y_start:y_end,
                x_start:x_end
            ]

            actual_h = y_end - y_start
            actual_w = x_end - x_start

            if (
                actual_h < tile_size
                or actual_w < tile_size
            ):

                tile_padded = np.zeros(
                    (
                        tile_size,
                        tile_size,
                        3
                    ),
                    dtype=np.uint8
                )

                tile_padded[
                    :actual_h,
                    :actual_w
                ] = tile

                with torch.inference_mode():
                    tile_depth = depth_model.infer_image(
                        tile_padded,
                        tile_size
                    )

                tile_depth = tile_depth[
                    :actual_h,
                    :actual_w
                ]

            else:

                with torch.inference_mode():
                    tile_depth = depth_model.infer_image(
                        tile,
                        tile_size
                    )

            weight = np.ones(
                (
                    actual_h,
                    actual_w
                ),
                dtype=np.float32
            )

            if overlap > 0:

                if i > 0:

                    fade = np.linspace(
                        0,
                        1,
                        overlap
                    )

                    weight[
                        :overlap,
                        :
                    ] *= fade[:, None]

                if i < n_tiles_h - 1:

                    fade = np.linspace(
                        1,
                        0,
                        overlap
                    )

                    weight[
                        -overlap:,
                        :
                    ] *= fade[:, None]

                if j > 0:

                    fade = np.linspace(
                        0,
                        1,
                        overlap
                    )

                    weight[
                        :,
                        :overlap
                    ] *= fade[None, :]

                if j < n_tiles_w - 1:

                    fade = np.linspace(
                        1,
                        0,
                        overlap
                    )

                    weight[
                        :,
                        -overlap:
                    ] *= fade[None, :]

            full_depth_map[
                y_start:y_end,
                x_start:x_end
            ] += (
                tile_depth * weight
            )

            weight_map[
                y_start:y_end,
                x_start:x_end
            ] += weight

            tile_counter += 1

            # Update the UI less frequently to reduce Streamlit overhead.
            if tile_counter == total_tiles or tile_counter % 2 == 0:
                progress.progress(
                    tile_counter / total_tiles,
                    text=(
                        f"Generating foreground... "
                        f"{tile_counter}/{total_tiles} tiles"
                    )
                )

            del tile_depth, weight


    full_depth_map /= np.maximum(
        weight_map,
        1e-8
    )

    full_depth_map = cv2.resize(
        full_depth_map,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    foreground = extract_foreground(
        raw_image,
        full_depth_map,
        method="adaptive",
        threshold=13.0,
        percentile=55,
        smooth_kernel=7
    )

    progress.empty()

    return foreground


# ============================================================
# YOLO BERRY DETECTION + ANNOTATED IMAGE
# ============================================================

def predict_berry_count(foreground, model, tile_size=640, conf=0.4):
    """CPU-safe YOLO tiling with only one tile resident at a time."""
    image = foreground
    original_h, original_w = image.shape[:2]
    output_image = image.copy()

    padded_w = math.ceil(original_w / tile_size) * tile_size
    padded_h = math.ceil(original_h / tile_size) * tile_size

    padded_image = cv2.copyMakeBorder(
        image, 0, padded_h - original_h, 0, padded_w - original_w,
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

                    if x1 >= original_w or y1 >= original_h or x2 <= 0 or y2 <= 0:
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
                    text=f"Detecting berries... {tile_counter}/{total_tiles} tiles"
                )

    progress.empty()
    del padded_image

    total_berries = total_green + total_ripe
    return total_green, total_ripe, total_berries, output_image


# ============================================================
# DINOv3 GENOTYPE PREDICTION
# ============================================================

def predict_genotype(
    foreground,
    processor,
    dino_model,
    genotype_model
):

    rgb = cv2.cvtColor(
        foreground,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        rgb
    )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    num_register_tokens = getattr(
        dino_model.config,
        "num_register_tokens",
        4
    )

    with torch.inference_mode():

        outputs = dino_model(
            **inputs
        )

        hidden = (
            outputs.last_hidden_state
        )

        patch_embeddings = hidden[
            :,
            1 + num_register_tokens:,
            :
        ]

        embedding = (
            patch_embeddings
            .mean(dim=1)
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Classifier prediction
    # --------------------------------------------------------

    genotype_id = (
        genotype_model
        .predict(embedding)[0]
    )

    genotype_id = int(
        genotype_id
    )

    # --------------------------------------------------------
    # Convert ID to genotype name
    # --------------------------------------------------------

    genotype_name = CLASS_NAMES.get(
        genotype_id,
        f"Unknown class ({genotype_id})"
    )

    return (
        genotype_id,
        genotype_name
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Settings")

    st.write(
        "AI models are stored on the "
        "application backend."
    )

    st.divider()

    st.write("**Foreground Extraction using:**")
    st.write("Depth Anything V2 ViT-S")

    st.write("**Blueberry detection model used to detect Blueberries based on Maturity:**")
    st.write("YOLO11l")

    st.write("**DINOv3 Vision Foundation model used to estimate Plant Yield Potential:**")
    st.write("DINOv3 ViT-B/16")

    st.write("**Blueberry Yield Estimation for a plant (in g):**")
    st.write("ElasticNet Regressor")

    st.divider()

    st.write(
        f"Compute device: `{DEVICE}`"
    )
    st.caption(
        "CPU-optimized configuration for Streamlit Cloud"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload Blueberry Image")

uploaded_file = st.file_uploader(
    "Choose an original blueberry plant image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read original image
    # --------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Original Image"
    )

    st.image(
        original_image,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # User input
    # --------------------------------------------------------

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

        temp_path = None
        depth_model = None
        yolo_model = None
        processor = None
        dino_model = None

        try:

            # =================================================
            # SAVE UPLOADED IMAGE TEMPORARILY
            # =================================================

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            # =================================================
            # 1. FOREGROUND GENERATION
            # =================================================

            st.subheader("2️⃣ Foreground Extraction")

            with st.spinner(
                "Loading Depth Anything V2 on CPU and generating foreground..."
            ):
                depth_model = load_depth_model()

                foreground = generate_foreground(
                    temp_path,
                    depth_model
                )

            # Release the largest model before loading YOLO.
            del depth_model
            depth_model = None
            gc.collect()

            foreground_rgb = cv2.cvtColor(
                foreground,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                foreground_rgb,
                use_container_width=True,
                caption="Generated foreground"
            )

            # =================================================
            # 2. BERRY DETECTION
            # =================================================

            st.subheader("3️⃣ Berry Detection")

            with st.spinner(
                "Loading YOLO and detecting green and ripe berries on CPU..."
            ):
                yolo_model = load_yolo_model()

                (
                    green,
                    ripe,
                    total,
                    annotated_image
                ) = predict_berry_count(
                    foreground,
                    yolo_model,
                    tile_size=640,
                    conf=0.4
                )

            # Release YOLO before loading DINO.
            del yolo_model
            yolo_model = None
            gc.collect()

            annotated_rgb = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                annotated_rgb,
                use_container_width=True,
                caption=(
                    ":black[**Detected berries** "
                    "(**Green** = green berries, "
                    "**Red** = ripe berries)]"
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🟢 Green Berries", green)

            with col2:
                st.metric("🔴 Ripe Berries", ripe)

            with col3:
                st.metric("🫐 Total Berries", total)

            # =================================================
            # 3. GENOTYPE PREDICTION
            # =================================================

            st.subheader("4️⃣ Genotype Prediction")

            with st.spinner(
                "Loading DINOv3 on CPU and predicting genotype..."
            ):
                processor, dino_model = load_dino_model()
                genotype_model = load_genotype_model()

                (
                    genotype_id,
                    genotype
                ) = predict_genotype(
                    foreground,
                    processor,
                    dino_model,
                    genotype_model
                )

            # Release DINO immediately after feature extraction.
            del processor
            del dino_model
            processor = None
            dino_model = None
            gc.collect()

            # Foreground is no longer needed after DINO feature extraction.
            # Keep the annotated image, but release the large source array.
            del foreground
            gc.collect()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Genotype",
                    genotype
                )

            if genotype_id <= 7:
                yield_text = "Low yield potential"
            elif genotype_id <= 12:
                yield_text = "Average yield potential"
            else:
                yield_text = "High yield potential"

            with col2:
                st.metric(
                    label="Yield Potential Estimation",
                    value=yield_text
                )

            # =================================================
            # 4. YIELD PREDICTION
            # =================================================

            st.subheader("5️⃣ Yield Prediction")

            with st.spinner("Estimating blueberry yield..."):

                yield_model = load_yield_model()

                X_new = pd.DataFrame({
                    "Genotype": [genotype_id],
                    "Avg. Berry wt": [berry_weight],
                    "DL berry count": [ripe]
                })

                predicted_yield = float(
                    yield_model.predict(X_new)[0]
                )
                del yield_model
                gc.collect()

            # =================================================
            # FINAL YIELD
            # =================================================

            st.success(
                "Analysis completed successfully!"
            )

            st.divider()

            st.markdown(
                """
                <div class="result-box">
                    <div class="metric-title">
                        ESTIMATED BLUEBERRY YIELD
                    </div>
                    <div class="yield-value">
                        {:.2f} g
                    </div>
                </div>
                """.format(predicted_yield),
                unsafe_allow_html=True
            )

            st.divider()

            # =================================================
            # SUMMARY
            # =================================================

            st.subheader("📋 Prediction Summary")

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
                    ripe,
                    green,
                    total,
                    yield_text,
                    f"{predicted_yield:.2f} g"
                ]
            })

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "An error occurred during analysis."
            )
            st.exception(e)

        finally:

            # Always release model memory and temporary files.
            depth_model = None
            yolo_model = None
            processor = None
            dino_model = None

            gc.collect()

            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

else:

    st.info(
        "Upload an original blueberry plant image "
        "to begin the analysis."
    )
