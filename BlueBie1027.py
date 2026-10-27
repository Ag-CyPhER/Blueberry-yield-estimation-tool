import os
import math
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO

# ------------------------------------------------------------------
# FOREGROUND EXTRACTION IS DISABLED (all related code commented out)
# ------------------------------------------------------------------
# from depth_anything_v2.dpt import DepthAnythingV2
# from huggingface_hub import hf_hub_download

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

# --- Depth model (foreground extraction) - disabled ---
# DEPTH_REPO_ID = "puranjit13/depthanythingv2_vkitti"
# DEPTH_FILENAME = "depth_anything_v2_metric_vkitti_vitl.pth"

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

GENOTYPE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "DINOv3_vitb16_genotype_classifier.pkl"
)

YIELD_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "DINOv3_yield_ElasticNet.pkl"
)

DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# Longest image side (px) used for the interactive zoom viewer.
# Display only - never affects the analysis.
DISPLAY_MAX_SIDE = 2000


# ============================================================
# DEVICE (CPU only)
# ============================================================

DEVICE = torch.device("cpu")

# Optional: use all available CPU cores for inference
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

    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f4f8f4;
        border: 1px solid #d8e5d8;
        text-align: center;
    }

    .yield-value {
        font-size: clamp(28px, 6vw, 38px);
        font-weight: 700;
    }

    .metric-title {
        font-size: 16px;
        color: #666;
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
    '<div class="main-title">🫐 Blueberry Yield Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based blueberry genotype, blueberry detection, and yield estimation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DEPTH ANYTHING V2  (DISABLED - foreground extraction)
# ============================================================

# @st.cache_resource
# def load_depth_model():
#
#     model_configs = {
#         "vitl": {
#             "encoder": "vitl",
#             "features": 256,
#             "out_channels": [256, 512, 1024, 1024]
#         }
#     }
#
#     depth_checkpoint = hf_hub_download(
#         repo_id=DEPTH_REPO_ID,
#         filename=DEPTH_FILENAME
#     )
#
#     model = DepthAnythingV2(
#         **model_configs["vitl"],
#         max_depth=75
#     )
#
#     model.load_state_dict(
#         torch.load(depth_checkpoint, map_location="cpu")
#     )
#
#     model = model.to(DEVICE).eval()
#
#     return model


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_yolo_model():

    return YOLO(YOLO_MODEL_PATH)


# ============================================================
# LOAD DINOv3
# ============================================================

@st.cache_resource
def load_dino_model():

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)

    model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE)

    model.eval()

    return processor, model


# ============================================================
# LOAD GENOTYPE CLASSIFIER
# ============================================================

@st.cache_resource
def load_genotype_model():

    return joblib.load(GENOTYPE_MODEL_PATH)


# ============================================================
# LOAD YIELD MODEL
# ============================================================

@st.cache_resource
def load_yield_model():

    return joblib.load(YIELD_MODEL_PATH)


# ============================================================
# FOREGROUND EXTRACTION  (DISABLED)
# ============================================================

# def extract_foreground(
#     image,
#     depth_map,
#     method="adaptive",
#     threshold=None,
#     percentile=55,
#     smooth_kernel=7
# ):
#
#     if method == "percentile":
#         depth_threshold = np.percentile(depth_map, percentile)
#         mask = depth_map <= depth_threshold
#
#     elif method == "absolute":
#         if threshold is None:
#             threshold = np.mean(depth_map) * 0.5
#         mask = depth_map <= threshold
#
#     elif method == "adaptive":
#         depth_normalized = (
#             (depth_map - depth_map.min()) /
#             (depth_map.max() - depth_map.min() + 1e-8) * 255
#         ).astype(np.uint8)
#
#         _, mask = cv2.threshold(
#             depth_normalized, 0, 255,
#             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#         )
#         mask = mask > 0
#
#     else:
#         raise ValueError("Unknown foreground method")
#
#     if smooth_kernel > 0:
#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (smooth_kernel, smooth_kernel)
#         )
#         mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = mask > 0
#
#     foreground = image.copy()
#     foreground[~mask] = 0
#
#     return foreground


# ============================================================
# GENERATE FOREGROUND  (DISABLED)
# ============================================================

# def generate_foreground(image_path, depth_model):
#
#     raw_image = cv2.imread(image_path)
#
#     if raw_image is None:
#         raise ValueError("Could not read uploaded image.")
#
#     orig_h, orig_w = raw_image.shape[:2]
#
#     input_size = 512
#     overlap = 64
#     tile_size = input_size
#     stride = tile_size - overlap
#
#     new_h = ((orig_h - overlap + stride - 1) // stride) * stride + overlap
#     new_w = ((orig_w - overlap + stride - 1) // stride) * stride + overlap
#
#     resized_image = cv2.resize(
#         raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
#     )
#
#     n_tiles_h = (new_h - overlap) // stride
#     n_tiles_w = (new_w - overlap) // stride
#
#     full_depth_map = np.zeros((new_h, new_w), dtype=np.float32)
#     weight_map = np.zeros((new_h, new_w), dtype=np.float32)
#
#     progress = st.progress(0, text="Generating foreground...")
#     total_tiles = n_tiles_h * n_tiles_w
#     tile_counter = 0
#
#     for i in range(n_tiles_h):
#         for j in range(n_tiles_w):
#
#             y_start = i * stride
#             y_end = min(y_start + tile_size, new_h)
#             x_start = j * stride
#             x_end = min(x_start + tile_size, new_w)
#
#             tile = resized_image[y_start:y_end, x_start:x_end]
#             actual_h = y_end - y_start
#             actual_w = x_end - x_start
#
#             if actual_h < tile_size or actual_w < tile_size:
#                 tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
#                 tile_padded[:actual_h, :actual_w] = tile
#                 tile_depth = depth_model.infer_image(tile_padded, tile_size)
#                 tile_depth = tile_depth[:actual_h, :actual_w]
#             else:
#                 tile_depth = depth_model.infer_image(tile, tile_size)
#
#             weight = np.ones((actual_h, actual_w), dtype=np.float32)
#
#             if overlap > 0:
#                 if i > 0:
#                     fade = np.linspace(0, 1, overlap)
#                     weight[:overlap, :] *= fade[:, None]
#                 if i < n_tiles_h - 1:
#                     fade = np.linspace(1, 0, overlap)
#                     weight[-overlap:, :] *= fade[:, None]
#                 if j > 0:
#                     fade = np.linspace(0, 1, overlap)
#                     weight[:, :overlap] *= fade[None, :]
#                 if j < n_tiles_w - 1:
#                     fade = np.linspace(1, 0, overlap)
#                     weight[:, -overlap:] *= fade[None, :]
#
#             full_depth_map[y_start:y_end, x_start:x_end] += tile_depth * weight
#             weight_map[y_start:y_end, x_start:x_end] += weight
#
#             tile_counter += 1
#             progress.progress(
#                 tile_counter / total_tiles,
#                 text=f"Generating foreground... {tile_counter}/{total_tiles} tiles"
#             )
#
#     full_depth_map /= np.maximum(weight_map, 1e-8)
#
#     full_depth_map = cv2.resize(
#         full_depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
#     )
#
#     foreground = extract_foreground(
#         raw_image, full_depth_map,
#         method="adaptive", threshold=13.0,
#         percentile=55, smooth_kernel=7
#     )
#
#     progress.empty()
#
#     return foreground


# ============================================================
# YOLO BERRY DETECTION + ANNOTATED IMAGE
# ============================================================

def predict_berry_count(
    image_bgr,
    model,
    tile_size=640,
    conf=0.4
):

    """
    Runs YOLO (on CPU) directly on the ORIGINAL uploaded image.

    Green berries -> green boxes
    Ripe berries  -> red boxes

    Returns:
        total_green, total_ripe, total_berries, annotated_image (BGR)
    """

    image = image_bgr.copy()

    original_h, original_w = image.shape[:2]

    output_image = image.copy()

    padded_w = math.ceil(original_w / tile_size) * tile_size
    padded_h = math.ceil(original_h / tile_size) * tile_size

    padded_image = cv2.copyMakeBorder(
        image,
        0,
        padded_h - original_h,
        0,
        padded_w - original_w,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    tiles = []
    tile_positions = []

    for y in range(0, padded_h, tile_size):
        for x in range(0, padded_w, tile_size):

            tile = padded_image[y:y + tile_size, x:x + tile_size]

            tiles.append(tile)
            tile_positions.append((x, y))

    # Run YOLO on CPU, in small batches to keep memory low
    results = []
    batch_size = 8

    progress = st.progress(0, text="Detecting blueberries...")

    for b in range(0, len(tiles), batch_size):

        results.extend(
            model.predict(
                source=tiles[b:b + batch_size],
                conf=conf,
                imgsz=tile_size,
                device="cpu",
                verbose=False
            )
        )

        progress.progress(
            min(1.0, (b + batch_size) / len(tiles)),
            text=f"Detecting blueberries... "
                 f"{min(b + batch_size, len(tiles))}/{len(tiles)} tiles"
        )

    progress.empty()

    total_green = 0
    total_ripe = 0

    for result, (tile_x, tile_y) in zip(results, tile_positions):

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, class_id in zip(boxes, class_ids):

            class_name = result.names[class_id]
            class_name_lower = class_name.lower()

            if class_name_lower == "green":
                total_green += 1
                color = (0, 255, 0)      # BGR green

            elif class_name_lower == "ripe":
                total_ripe += 1
                color = (0, 0, 255)      # BGR red

            else:
                continue

            x1, y1, x2, y2 = box

            x1 = int(x1 + tile_x)
            y1 = int(y1 + tile_y)
            x2 = int(x2 + tile_x)
            y2 = int(y2 + tile_y)

            if x1 >= original_w or y1 >= original_h:
                continue

            if x2 <= 0 or y2 <= 0:
                continue

            x1 = max(0, min(x1, original_w - 1))
            y1 = max(0, min(y1, original_h - 1))
            x2 = max(0, min(x2, original_w - 1))
            y2 = max(0, min(y2, original_h - 1))

            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                output_image,
                class_name,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    total_berries = total_green + total_ripe

    return (
        total_green,
        total_ripe,
        total_berries,
        output_image
    )


# ============================================================
# DINOv3 GENOTYPE PREDICTION (runs on the original image, CPU)
# ============================================================

def predict_genotype(
    image_bgr,
    processor,
    dino_model,
    genotype_model
):

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt")

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    num_register_tokens = getattr(
        dino_model.config,
        "num_register_tokens",
        4
    )

    with torch.inference_mode():

        outputs = dino_model(**inputs)

        hidden = outputs.last_hidden_state

        patch_embeddings = hidden[:, 1 + num_register_tokens:, :]

        embedding = patch_embeddings.mean(dim=1).cpu().numpy()

    genotype_id = int(genotype_model.predict(embedding)[0])

    genotype_name = CLASS_NAMES.get(
        genotype_id,
        f"Unknown class ({genotype_id})"
    )

    return genotype_id, genotype_name


# ============================================================
# ZOOMABLE 2x1 (VERTICAL) COMPARISON FIGURE  (plotly subplots)
# ============================================================

def _shrink_for_display(img_rgb, max_side=DISPLAY_MAX_SIDE):

    h, w = img_rgb.shape[:2]

    scale = max_side / max(h, w)

    if scale >= 1:
        return img_rgb

    return cv2.resize(
        img_rgb,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def build_zoom_figure(original_rgb, annotated_rgb):
    """
    Two stacked panels (2 rows x 1 column):
        row 1: original image
        row 2: image with berry detections

    Each panel zooms / pans independently:
        - drag a box to zoom into a region
        - mouse wheel to zoom at the cursor
        - double-click to reset the view
    """

    top = _shrink_for_display(original_rgb)
    bottom = _shrink_for_display(annotated_rgb)

    h, w = top.shape[:2]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "1. Original Image",
            "2. Detected Blueberries"
        ),
        vertical_spacing=0.06
    )

    # binary_string=True embeds the image as one compact PNG
    fig.add_trace(
        px.imshow(top, binary_string=True).data[0],
        row=1, col=1
    )

    fig.add_trace(
        px.imshow(bottom, binary_string=True).data[0],
        row=2, col=1
    )

    for r in (1, 2):

        x_id = "x" if r == 1 else f"x{r}"

        fig.update_xaxes(
            visible=False,
            range=[-0.5, w - 0.5],
            row=r, col=1
        )

        # Reversed y-axis (image coordinates) + locked aspect ratio
        fig.update_yaxes(
            visible=False,
            autorange=False,
            range=[h - 0.5, -0.5],
            scaleanchor=x_id,
            row=r, col=1
        )

    # Legend (colour key for the detection panel)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="lime", width=3),
            name="Immature blueberries"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="red", width=3),
            name="Ripe blueberries"
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=int(min(1800, max(900, 2 * 700 * h / w))),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.01,
            xanchor="center", x=0.5
        ),
        dragmode="zoom"
    )

    fig.update_annotations(font_size=18)

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.divider()

    st.write("**Blueberry detection based on maturity:**")
    st.write("YOLO11l")

    st.write("**Vision Foundation model for Yield Potential Estimation:**")
    st.write("DINOv3 ViT-B/16")

    st.write("**Yield Estimation model:**")
    st.write("ElasticNet Regressor")

    st.divider()

    st.write(f"Inference running on `{DEVICE}`")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload Blueberry Image")

st.write(
    "Recommendation: Please upload an image of a single, cropped "
    "blueberry plant for a reliable blueberry detection count and "
    "yield estimation!!"
)

uploaded_file = st.file_uploader(
    "Choose an original blueberry plant image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    for _k in ("upload_id", "detection", "final"):
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

    # --------------------------------------------------------
    # STAGE 1 (automatic on upload):
    # blueberry detection on the ORIGINAL image -> figure -> counts
    # --------------------------------------------------------

    if st.session_state.get("upload_id") != upload_id:

        st.session_state["upload_id"] = upload_id
        st.session_state.pop("detection", None)
        st.session_state.pop("final", None)

        try:

            with st.spinner("Loading AI models..."):

                # depth_model = load_depth_model()   # foreground disabled
                yolo_model = load_yolo_model()

            # Decode the uploaded image directly (no temp file needed)
            file_bytes = np.frombuffer(
                uploaded_file.getbuffer(),
                dtype=np.uint8
            )

            original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if original_bgr is None:
                raise ValueError("Could not read uploaded image.")

            # ----------------------------------------------------
            # FOREGROUND EXTRACTION STEP (DISABLED)
            # ----------------------------------------------------
            # foreground = generate_foreground(temp_path, depth_model)
            # The original image is used everywhere instead.

            with st.spinner(
                "Detecting immature and ripe blueberries "
                "(CPU - this may take a moment)..."
            ):

                (
                    green,
                    ripe,
                    total,
                    annotated_image
                ) = predict_berry_count(
                    original_bgr,
                    yolo_model,
                    tile_size=640,
                    conf=0.4
                )

            with st.spinner("Preparing visualization..."):

                fig = build_zoom_figure(
                    cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                )

            st.session_state["detection"] = {
                "image_bgr": original_bgr,   # needed for genotype step
                "figure": fig,
                "green": green,
                "ripe": ripe,
                "total": total,
            }

        except Exception as e:

            st.error("An error occurred during image analysis.")

            st.exception(e)

    detection = st.session_state.get("detection")

    if detection is not None:

        # ----------------------------------------------------
        # Visualization: Original (top) / Detections (bottom)
        # ----------------------------------------------------

        st.subheader("🔍 Image Analysis")

        st.caption(
            "Zoom: drag a box over a region, or scroll the mouse wheel. "
            "Pan: hold and drag after choosing the pan tool in the toolbar. "
            "Double-click an image to reset its view."
        )

        st.plotly_chart(
            detection["figure"],
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            }
        )

        # ----------------------------------------------------
        # Blueberry detection counts
        # ----------------------------------------------------

        st.subheader("🫐 Blueberry Detection Count")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🟢 Immature Blueberries", detection["green"])

        with col2:
            st.metric("🔴 Ripe Blueberries", detection["ripe"])

        with col3:
            st.metric("🫐 Total Blueberries", detection["total"])

        st.divider()

        # ----------------------------------------------------
        # STAGE 2 (on user input):
        # average berry weight -> genotype + yield potential -> yield
        # ----------------------------------------------------

        st.subheader("⚖️ Yield Estimation")

        st.write("Enter an average blueberry weight")

        inp_col, btn_col = st.columns(
            [2, 1],
            vertical_alignment="bottom"
        )

        with inp_col:

            berry_weight = st.number_input(
                "Avg. Berry wt ",
                min_value=0.10,
                max_value=10.0,
                value=2.0,
                step=0.01
            )

        with btn_col:

            estimate_button = st.button(
                "Estimate Genotype & Yield",
                type="primary",
                use_container_width=True
            )

        if estimate_button:

            try:

                with st.spinner("Loading AI models..."):

                    processor, dino_model = load_dino_model()
                    genotype_model = load_genotype_model()
                    yield_model = load_yield_model()

                with st.spinner(
                    "Extracting DINOv3 features "
                    "and predicting genotype..."
                ):

                    (
                        genotype_id,
                        genotype
                    ) = predict_genotype(
                        detection["image_bgr"],
                        processor,
                        dino_model,
                        genotype_model
                    )

                if genotype_id <= 7:
                    yield_text = "Low yield potential"
                elif 7 < genotype_id <= 12:
                    yield_text = "Medium yield potential"
                else:
                    yield_text = "High yield potential"

                with st.spinner("Estimating blueberry yield..."):

                    # The DL berry count used by the ElasticNet model
                    # is total_ripe.
                    X_new = pd.DataFrame({
                        "Genotype": [genotype_id],
                        "Avg. Berry wt": [berry_weight],
                        "DL berry count": [detection["ripe"]]
                    })

                    predicted_yield = yield_model.predict(X_new)[0]

                st.session_state["final"] = {
                    "genotype_id": genotype_id,
                    "genotype": genotype,
                    "yield_text": yield_text,
                    "berry_weight": berry_weight,
                    "predicted_yield": predicted_yield,
                }

            except Exception as e:

                st.error("An error occurred during yield estimation.")

                st.exception(e)

        final = st.session_state.get("final")

        if final is not None:

            st.subheader("🧬 Genotype Prediction")

            g_col1, g_col2 = st.columns(2)

            with g_col1:
                st.metric("Predicted Genotype", final["genotype"])

            with g_col2:
                st.metric(
                    label="Yield Potential Estimation",
                    value=final["yield_text"]
                )

            st.success("Analysis completed successfully!")

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
                """.format(final["predicted_yield"]),
                unsafe_allow_html=True
            )

            st.divider()

            st.subheader("📋 Prediction Summary")

            summary = pd.DataFrame({
                "Parameter": [
                    "Predicted Genotype",
                    "Avg. berry wt",
                    "DL Ripe Blueberry Count",
                    "Immature Blueberry Count",
                    "Total Blueberry Count",
                    "Yield Potential Estimation",
                    "Predicted Yield"
                ],
                "Value": [
                    final["genotype"],
                    f"{final['berry_weight']:.2f} g",
                    detection["ripe"],
                    detection["green"],
                    detection["total"],
                    final["yield_text"],
                    f"{final['predicted_yield']:.2f} g"
                ]
            })

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True
            )
