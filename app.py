import os
import math
import tempfile
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
import streamlit as st
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

# DEPTH_CHECKPOINT = os.path.join(
#     MODEL_DIR,
#     "depth_anything_v2_metric_vkitti_vitl.pth"
# )

from huggingface_hub import hf_hub_download

DEPTH_REPO_ID = "puranjit13/depthanythingv2_vkitti"
DEPTH_FILENAME = "depth_anything_v2_metric_vkitti_vitl.pth"

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
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


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

# @st.cache_resource
# def load_depth_model():

#     model_configs = {
#         "vitl": {
#             "encoder": "vitl",
#             "features": 256,
#             "out_channels": [
#                 256,
#                 512,
#                 1024,
#                 1024
#             ]
#         }
#     }

#     model = DepthAnythingV2(
#         **model_configs["vitl"],
#         max_depth=75
#     )

#     model.load_state_dict(
#         torch.load(
#             DEPTH_CHECKPOINT,
#             map_location="cpu"
#         )
#     )

#     model = model.to(DEVICE).eval()

#     return model

@st.cache_resource
def load_depth_model():

    model_configs = {
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [
                256,
                512,
                1024,
                1024
            ]
        }
    }

    # Download checkpoint from Hugging Face
    depth_checkpoint = hf_hub_download(
        repo_id=DEPTH_REPO_ID,
        filename=DEPTH_FILENAME
    )

    model = DepthAnythingV2(
        **model_configs["vitl"],
        max_depth=75
    )

    model.load_state_dict(
        torch.load(
            depth_checkpoint,
            map_location="cpu"
        )
    )

    model = model.to(DEVICE).eval()

    return model

# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_yolo_model():

    return YOLO(
        YOLO_MODEL_PATH
    )


# ============================================================
# LOAD DINOv3
# ============================================================

@st.cache_resource
def load_dino_model():

    processor = AutoImageProcessor.from_pretrained(
        DINO_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        DINO_MODEL_NAME
    ).to(DEVICE)

    model.eval()

    return processor, model


# ============================================================
# LOAD GENOTYPE CLASSIFIER
# ============================================================

@st.cache_resource
def load_genotype_model():

    return joblib.load(
        GENOTYPE_MODEL_PATH
    )


# ============================================================
# LOAD YIELD MODEL
# ============================================================

@st.cache_resource
def load_yield_model():

    return joblib.load(
        YIELD_MODEL_PATH
    )


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

    input_size = 512
    overlap = 64

    tile_size = input_size
    stride = tile_size - overlap

    new_h = (
        (
            orig_h
            - overlap
            + stride
            - 1
        )
        // stride
    ) * stride + overlap

    new_w = (
        (
            orig_w
            - overlap
            + stride
            - 1
        )
        // stride
    ) * stride + overlap

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

                tile_depth = (
                    depth_model.infer_image(
                        tile_padded,
                        tile_size
                    )
                )

                tile_depth = tile_depth[
                    :actual_h,
                    :actual_w
                ]

            else:

                tile_depth = (
                    depth_model.infer_image(
                        tile,
                        tile_size
                    )
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

            progress.progress(
                tile_counter / total_tiles,
                text=(
                    f"Generating foreground... "
                    f"{tile_counter}/{total_tiles} tiles"
                )
            )

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

def predict_berry_count(
    foreground,
    model,
    tile_size=640,
    conf=0.4
):

    """
    Runs YOLO on the generated foreground image.

    Green berries:
        Green bounding boxes

    Ripe berries:
        Red bounding boxes

    Returns:
        total_green
        total_ripe
        total_berries
        annotated_image
    """

    image = foreground.copy()

    original_h, original_w = image.shape[:2]

    # --------------------------------------------------------
    # Copy for drawing detections
    # --------------------------------------------------------

    output_image = image.copy()

    # --------------------------------------------------------
    # Calculate padded dimensions
    # --------------------------------------------------------

    padded_w = (
        math.ceil(
            original_w / tile_size
        ) * tile_size
    )

    padded_h = (
        math.ceil(
            original_h / tile_size
        ) * tile_size
    )

    # --------------------------------------------------------
    # Pad foreground
    # --------------------------------------------------------

    padded_image = cv2.copyMakeBorder(
        image,
        0,
        padded_h - original_h,
        0,
        padded_w - original_w,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # --------------------------------------------------------
    # Create tiles and remember positions
    # --------------------------------------------------------

    tiles = []
    tile_positions = []

    for y in range(
        0,
        padded_h,
        tile_size
    ):

        for x in range(
            0,
            padded_w,
            tile_size
        ):

            tile = padded_image[
                y:y + tile_size,
                x:x + tile_size
            ]

            tiles.append(tile)
            tile_positions.append(
                (x, y)
            )

    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    results = model.predict(
        source=tiles,
        conf=conf,
        imgsz=tile_size,
        verbose=False
    )

    # --------------------------------------------------------
    # Count detections
    # --------------------------------------------------------

    total_green = 0
    total_ripe = 0

    for result, (
        tile_x,
        tile_y
    ) in zip(
        results,
        tile_positions
    ):

        if result.boxes is None:
            continue

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        class_ids = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, class_id in zip(
            boxes,
            class_ids
        ):

            class_name = (
                result.names[class_id]
            )

            class_name_lower = (
                class_name.lower()
            )

            # ------------------------------------------------
            # Green berries
            # ------------------------------------------------

            if class_name_lower == "green":

                total_green += 1

                # BGR = Green
                color = (
                    0,
                    255,
                    0
                )

            # ------------------------------------------------
            # Ripe berries
            # ------------------------------------------------

            elif class_name_lower == "ripe":

                total_ripe += 1

                # BGR = Red
                color = (
                    0,
                    0,
                    255
                )

            else:

                continue

            # ------------------------------------------------
            # Convert tile coordinates to original image
            # coordinates
            # ------------------------------------------------

            x1, y1, x2, y2 = box

            x1 = int(
                x1 + tile_x
            )

            y1 = int(
                y1 + tile_y
            )

            x2 = int(
                x2 + tile_x
            )

            y2 = int(
                y2 + tile_y
            )

            # ------------------------------------------------
            # Ignore detections outside original image
            # ------------------------------------------------

            if (
                x1 >= original_w
                or y1 >= original_h
            ):
                continue

            if (
                x2 <= 0
                or y2 <= 0
            ):
                continue

            # ------------------------------------------------
            # Clip bounding boxes
            # ------------------------------------------------

            x1 = max(
                0,
                min(
                    x1,
                    original_w - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    original_h - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    original_w - 1
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    original_h - 1
                )
            )

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                output_image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # ------------------------------------------------
            # Draw label
            # ------------------------------------------------

            label = class_name

            cv2.putText(
                output_image,
                label,
                (
                    x1,
                    max(
                        y1 - 8,
                        15
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # --------------------------------------------------------
    # Total berries
    # --------------------------------------------------------

    total_berries = (
        total_green
        + total_ripe
    )

    return (
        total_green,
        total_ripe,
        total_berries,
        output_image
    )


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

    st.write("**Depth model to extract Foreground:**")
    st.write("Depth Anything V2 ViT-L")

    st.write("**Blueberry detector based on maturity:**")
    st.write("YOLO11l")

    st.write("**Vision Foundation model for Yield Potential Estimation:**")
    st.write("DINOv3 ViT-B/16")

    st.write("**Yield Estimation model:**")
    st.write("ElasticNet")

    st.divider()

    st.write(
        # f"Compute device: `{DEVICE}`"
        f"Inference running on NVIDIA GeForce RTX 5070Ti `{DEVICE}`"
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

        try:

            # =================================================
            # LOAD MODELS
            # =================================================

            with st.spinner(
                "Loading AI models..."
            ):

                depth_model = (
                    load_depth_model()
                )

                yolo_model = (
                    load_yolo_model()
                )

                processor, dino_model = (
                    load_dino_model()
                )

                genotype_model = (
                    load_genotype_model()
                )

                yield_model = (
                    load_yield_model()
                )

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

                tmp.write(
                    uploaded_file.getbuffer()
                )

                temp_path = tmp.name

            # =================================================
            # FOREGROUND GENERATION
            # =================================================

            st.subheader(
                "2️⃣ Foreground Extraction"
            )

            foreground = (
                generate_foreground(
                    temp_path,
                    depth_model
                )
            )

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
            # BERRY DETECTION
            # =================================================

            st.subheader(
                "3️⃣ Berry Detection"
            )

            with st.spinner(
                "Detecting green and ripe berries..."
            ):

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

            # ------------------------------------------------
            # Display annotated foreground
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Berry metrics
            # ------------------------------------------------

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "🟢 Green Berries",
                    green
                )

            with col2:

                st.metric(
                    "🔴 Ripe Berries",
                    ripe
                )

            with col3:

                st.metric(
                    "🫐 Total Berries",
                    total
                )

            # =================================================
            # GENOTYPE PREDICTION
            # =================================================

            st.subheader(
                "4️⃣ Genotype Prediction"
            )

            with st.spinner(
                "Extracting DINOv3 features "
                "and predicting genotype..."
            ):

                (
                    genotype_id,
                    genotype
                ) = predict_genotype(
                    foreground,
                    processor,
                    dino_model,
                    genotype_model
                )

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                st.metric(
                    "Predicted Genotype",
                    genotype
                )

            # with col2:

            #     st.metric(
            #         "Classifier Class ID",
            #         genotype_id
            #     )

            # 1. Determine the yield potential text first
            if genotype_id <= 7:
                yield_text = "Low yield potential"
            elif 7 < genotype_id <= 12:
                yield_text = "Average yield potential"
            else:
                yield_text = "High yield potential"

            # 2. Pass the variable into st.metric inside the column block
            with col2:
                st.metric(label="Yield Potential Estimation", value=yield_text)
            # =================================================
            # YIELD PREDICTION
            # =================================================

            st.subheader(
                "5️⃣ Yield Prediction"
            )

            with st.spinner(
                "Estimating blueberry yield..."
            ):

                # IMPORTANT:
                # The DL berry count used by the
                # ElasticNet model is total_ripe.

                X_new = pd.DataFrame({
                    "Genotype": [genotype_id],
                    "Avg. Berry wt": [
                        berry_weight
                    ],
                    "DL berry count": [
                        ripe
                    ]
                })

                predicted_yield = (
                    yield_model
                    .predict(X_new)[0]
                )

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
                """.format(
                    predicted_yield
                ),
                unsafe_allow_html=True
            )

            st.divider()

            # =================================================
            # SUMMARY
            # =================================================

            st.subheader(
                "📋 Prediction Summary"
            )

            summary = pd.DataFrame({
                "Parameter": [
                    "Predicted Genotype",
                    # "Classifier Class ID",
                    "Average Berry Weight",
                    "DL Ripe Berry Count",
                    "Green Berry Count",
                    "Total Berry Count",
                    "Yield Potential Estimation",
                    "Predicted Yield"
                ],
                "Value": [
                    genotype,
                    # genotype_id,
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

            # =================================================
            # CLEANUP
            # =================================================

            try:

                os.remove(
                    temp_path
                )

            except:

                pass

        except Exception as e:

            st.error(
                "An error occurred during analysis."
            )

            st.exception(e)

else:

    st.info(
        "Upload an original blueberry plant image "
        "to begin the analysis."
    )