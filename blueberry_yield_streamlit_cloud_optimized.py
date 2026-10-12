# app.py

import os
import gc
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
import joblib


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Blueberry Yield Analyzer",
    page_icon="🫐",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

YOLO_MODEL_PATH = MODEL_DIR / "best.pt"
GENOTYPE_CLASSIFIER_PATH = MODEL_DIR / "DINOv3_vitb16_genotype_classifier.pkl"
YIELD_MODEL_PATH = MODEL_DIR / "DINOv3_yield_ElasticNet.pkl"

DEPTH_REPO_ID = "puranjit13/depthanythingv2_vkitti"
DEPTH_CHECKPOINT = "depth_anything_v2_metric_vkitti_vitl.pth"

DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# CPU optimization for Streamlit Cloud
CPU_THREADS = min(4, os.cpu_count() or 1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    torch.set_num_threads(CPU_THREADS)
except RuntimeError:
    pass

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# Genotype classifier mapping
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
    18: "Legacy",
}


# =========================================================
# 2. INITIALIZE SESSION STATE
# =========================================================

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None


# =========================================================
# 3. CACHE AND MEMORY MANAGEMENT
# =========================================================

def clear_app_memory():
    """
    Clear Streamlit model/data caches and stored prediction images.
    Models will load again when the next prediction is requested.
    """
    st.cache_data.clear()
    st.cache_resource.clear()

    st.session_state.prediction_results = None

    gc.collect()


# =========================================================
# 4. LOAD DEPTH ANYTHING V2
# =========================================================

@st.cache_resource(show_spinner="Loading Depth Anything V2...")
def load_depth_model():
    """
    Load Depth Anything V2 and download its checkpoint from Hugging Face.
    The checkpoint is cached by Hugging Face after its first download.

    The depth_anything_v2 package must be available in the project.
    """

    from depth_anything_v2.dpt import DepthAnythingV2

    checkpoint_path = hf_hub_download(
        repo_id=DEPTH_REPO_ID,
        filename=DEPTH_CHECKPOINT,
    )

    model_config = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "max_depth": 75,
    }

    model = DepthAnythingV2(**model_config)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # Support checkpoints saved either as a state dictionary
    # or inside a dictionary with a state_dict key.
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model = model.to("cpu")
    model.eval()

    return model


# =========================================================
# 5. LOAD YOLO MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading berry detection model...")
def load_yolo_model():
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}"
        )

    return YOLO(str(YOLO_MODEL_PATH))


# =========================================================
# 6. LOAD DINOv3 MODEL AND PROCESSOR
# =========================================================

@st.cache_resource(show_spinner="Loading DINOv3...")
def load_dinov3_model():
    processor = AutoImageProcessor.from_pretrained(
        DINO_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        DINO_MODEL_NAME
    )

    model = model.to("cpu")
    model.eval()

    return processor, model


# =========================================================
# 7. LOAD GENOTYPE CLASSIFIER
# =========================================================

@st.cache_resource(show_spinner="Loading genotype classifier...")
def load_genotype_classifier():
    if not GENOTYPE_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            "Genotype classifier not found: "
            f"{GENOTYPE_CLASSIFIER_PATH}"
        )

    return joblib.load(GENOTYPE_CLASSIFIER_PATH)


# =========================================================
# 8. LOAD YIELD PREDICTION MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading yield prediction model...")
def load_yield_model():
    if not YIELD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Yield model not found: {YIELD_MODEL_PATH}"
        )

    return joblib.load(YIELD_MODEL_PATH)


# =========================================================
# 9. DEPTH MAP GENERATION
# =========================================================

def get_tile_starts(length, tile_size=512, overlap=64):
    """
    Generate tile start positions while ensuring the final edge is covered.
    """

    if length <= tile_size:
        return [0]

    stride = tile_size - overlap

    starts = list(range(0, length - tile_size + 1, stride))

    final_start = length - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def generate_depth_map(
    image_bgr,
    depth_model,
    tile_size=512,
    overlap=64,
):
    """
    Predict a depth map tile by tile to limit peak memory usage.
    """

    height, width = image_bgr.shape[:2]

    depth_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    y_positions = get_tile_starts(
        height,
        tile_size,
        overlap,
    )

    x_positions = get_tile_starts(
        width,
        tile_size,
        overlap,
    )

    with torch.inference_mode():
        for y in y_positions:
            for x in x_positions:
                tile = image_bgr[
                    y:min(y + tile_size, height),
                    x:min(x + tile_size, width),
                ]

                tile_height, tile_width = tile.shape[:2]

                depth = depth_model.infer_image(
                    tile,
                    input_size=tile_size,
                )

                depth = np.asarray(
                    depth,
                    dtype=np.float32,
                )

                if depth.shape != (tile_height, tile_width):
                    depth = cv2.resize(
                        depth,
                        (tile_width, tile_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                depth_sum[
                    y:y + tile_height,
                    x:x + tile_width,
                ] += depth

                weight_sum[
                    y:y + tile_height,
                    x:x + tile_width,
                ] += 1.0

                del tile, depth

    weight_sum[weight_sum == 0] = 1.0

    depth_map = depth_sum / weight_sum

    del depth_sum, weight_sum
    gc.collect()

    return depth_map


# =========================================================
# 10. GENERATE FOREGROUND IMAGE
# =========================================================

def generate_foreground(
    image_bgr,
    depth_model,
    percentile=55,
    smooth_kernel=7,
):
    """
    Generate a foreground image using the predicted metric depth map.

    Pixels closer than the selected depth percentile are retained.
    The remaining background is replaced with white.
    """

    depth_map = generate_depth_map(
        image_bgr,
        depth_model,
        tile_size=512,
        overlap=64,
    )

    valid_depth = depth_map[
        np.isfinite(depth_map)
    ]

    if valid_depth.size == 0:
        raise ValueError(
            "Depth estimation did not return valid values."
        )

    threshold = np.percentile(
        valid_depth,
        percentile,
    )

    # For metric depth, smaller values generally indicate
    # objects closer to the camera.
    mask = (
        np.isfinite(depth_map)
        & (depth_map <= threshold)
    ).astype(np.uint8) * 255

    # Remove small isolated regions and smooth mask boundaries.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1

        mask = cv2.GaussianBlur(
            mask,
            (smooth_kernel, smooth_kernel),
            0,
        )

        _, mask = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY,
        )

    # Keep foreground pixels and whiten the background.
    foreground = np.full_like(
        image_bgr,
        255,
    )

    foreground[mask > 0] = image_bgr[mask > 0]

    del depth_map, valid_depth, mask
    gc.collect()

    return foreground


# =========================================================
# 11. YOLO BERRY DETECTION
# =========================================================

def predict_berry_count(
    image_bgr,
    model,
    tile_size=640,
    conf=0.4,
):
    """
    Detect berries on image tiles and draw bounding boxes.

    Green berries: green boxes
    Ripe berries: red boxes
    Half-ripe berries, if present: yellow boxes

    Returns:
        green_count
        ripe_count
        total_berries
        annotated_image
    """

    height, width = image_bgr.shape[:2]

    annotated_image = image_bgr.copy()

    green_count = 0
    ripe_count = 0
    total_berries = 0

    x_positions = get_tile_starts(
        width,
        tile_size,
        overlap=0,
    )

    y_positions = get_tile_starts(
        height,
        tile_size,
        overlap=0,
    )

    class_names = model.names

    for y in y_positions:
        for x in x_positions:

            tile = image_bgr[
                y:min(y + tile_size, height),
                x:min(x + tile_size, width),
            ]

            if tile.size == 0:
                continue

            results = model.predict(
                source=tile,
                conf=conf,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            if result.boxes is None:
                del tile, results
                continue

            boxes = result.boxes

            if len(boxes) == 0:
                del tile, results
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for box, class_id, confidence in zip(
                xyxy,
                class_ids,
                confidences,
            ):

                if isinstance(class_names, dict):
                    class_name = str(
                        class_names.get(class_id, class_id)
                    ).lower()
                else:
                    class_name = str(
                        class_names[class_id]
                    ).lower()

                x1, y1, x2, y2 = box.astype(int)

                # Convert tile coordinates to full-image coordinates.
                x1 += x
                x2 += x
                y1 += y
                y2 += y

                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                total_berries += 1

                if "ripe" in class_name:
                    ripe_count += 1
                    box_color = (0, 0, 255)

                elif "green" in class_name:
                    green_count += 1
                    box_color = (0, 255, 0)

                elif "half" in class_name:
                    box_color = (0, 255, 255)

                else:
                    box_color = (255, 0, 0)

                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                label = (
                    f"{class_name}: {confidence:.2f}"
                )

                cv2.putText(
                    annotated_image,
                    label,
                    (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

            del tile, results, boxes, xyxy, class_ids, confidences

    gc.collect()

    return (
        green_count,
        ripe_count,
        total_berries,
        annotated_image,
    )


# =========================================================
# 12. DINOv3 EMBEDDING EXTRACTION
# =========================================================

def extract_dinov3_embedding(
    image_bgr,
    processor,
    model,
):
    """
    Extract a mean patch-token embedding from DINOv3 ViT-B/16.

    The CLS token and register tokens are excluded from the mean.
    """

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(image_rgb)

    inputs = processor(
        images=pil_image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state

        num_register_tokens = getattr(
            model.config,
            "num_register_tokens",
            4,
        )

        patch_tokens = hidden_states[
            :,
            1 + num_register_tokens:,
            :,
        ]

        # Mean pooling over patch tokens.
        embedding = patch_tokens.mean(dim=1)

        embedding = (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    del image_rgb, pil_image, inputs, outputs
    del hidden_states, patch_tokens, embedding if False else image_bgr

    gc.collect()

    return embedding


# =========================================================
# 13. PREDICT GENOTYPE
# =========================================================

def predict_genotype(
    image_bgr,
    processor,
    dino_model,
    genotype_classifier,
):
    embedding = extract_dinov3_embedding(
        image_bgr,
        processor,
        dino_model,
    )

    prediction = genotype_classifier.predict(
        embedding.reshape(1, -1)
    )[0]

    # Handle classifiers trained with cultivar names.
    if isinstance(prediction, str):
        reverse_mapping = {
            name: class_id
            for class_id, name in CLASS_NAMES.items()
        }

        if prediction in reverse_mapping:
            genotype_id = reverse_mapping[prediction]
            genotype_name = prediction
        else:
            genotype_id = None
            genotype_name = prediction

    else:
        genotype_id = int(prediction)

        # Support a zero-based classifier if it was trained
        # using labels 0 through 17 instead of 1 through 18.
        if genotype_id not in CLASS_NAMES:
            if genotype_id + 1 in CLASS_NAMES:
                genotype_id += 1

        genotype_name = CLASS_NAMES.get(
            genotype_id,
            f"Unknown genotype ({genotype_id})",
        )

    del embedding
    gc.collect()

    return genotype_id, genotype_name


# =========================================================
# 14. PREDICT YIELD
# =========================================================

def predict_yield(
    yield_model,
    genotype_id,
    average_berry_weight,
    ripe_berry_count,
):
    """
    Predict yield using the same feature names expected by the model.
    """

    if genotype_id is None:
        raise ValueError(
            "The genotype classifier did not return a valid numeric "
            "genotype ID required by the yield model."
        )

    input_features = pd.DataFrame(
        [{
            "Genotype": int(genotype_id),
            "Avg. Berry wt": float(average_berry_weight),
            "DL berry count": int(ripe_berry_count),
        }]
    )

    predicted_yield = yield_model.predict(
        input_features
    )[0]

    predicted_yield = float(
        np.asarray(predicted_yield).reshape(-1)[0]
    )

    del input_features
    gc.collect()

    return predicted_yield


# =========================================================
# 15. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🫐 Blueberry Yield Analyzer")

    st.write(
        "Estimate blueberry yield using depth-based foreground "
        "extraction, YOLO berry detection, and DINOv3 genotype "
        "classification."
    )

    st.divider()

    st.subheader("Memory Management")

    st.caption(
        "Clear cached models and saved prediction results when you "
        "want to free memory. Models will reload for the next prediction."
    )

    if st.button(
        "🧹 Clear Cache & Memory",
        use_container_width=True,
        help="Clear model caches and the currently displayed prediction.",
    ):
        clear_app_memory()

        st.success(
            "Cache cleared. Models will reload when needed."
        )

        st.rerun()

    st.divider()

    st.caption(
        "For best results, use a clear image of an individual blueberry "
        "plant with visible berries."
    )


# =========================================================
# 16. MAIN UI
# =========================================================

st.title("🫐 Blueberry Yield Analyzer")

st.write(
    "Upload a blueberry plant image to extract the foreground, "
    "detect berries, identify the predicted genotype, and estimate yield."
)

uploaded_file = st.file_uploader(
    "Upload a blueberry plant image",
    type=["jpg", "jpeg", "png"],
    key="uploaded_image",
)

average_berry_weight = st.number_input(
    "Average berry weight (g)",
    min_value=0.01,
    value=2.0,
    step=0.1,
    format="%.2f",
    help="Enter the average weight of an individual ripe berry in grams.",
)

run_button = st.button(
    "🔍 Analyze Image & Predict Yield",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)


# =========================================================
# 17. RUN THE PIPELINE
# =========================================================

if run_button and uploaded_file is not None:

    st.session_state.prediction_results = None

    try:
        with st.spinner(
            "Analyzing image. This may take a few minutes on CPU..."
        ):

            # -------------------------------------------------
            # Decode uploaded image
            # -------------------------------------------------

            pil_image = Image.open(
                uploaded_file
            ).convert("RGB")

            original_rgb = np.array(
                pil_image
            )

            original_bgr = cv2.cvtColor(
                original_rgb,
                cv2.COLOR_RGB2BGR,
            )

            del pil_image
            gc.collect()

            # -------------------------------------------------
            # Load cached models
            # -------------------------------------------------

            depth_model = load_depth_model()
            yolo_model = load_yolo_model()

            processor, dino_model = load_dinov3_model()

            genotype_classifier = load_genotype_classifier()
            yield_model = load_yield_model()

            # -------------------------------------------------
            # Generate foreground
            # -------------------------------------------------

            foreground_bgr = generate_foreground(
                original_bgr,
                depth_model,
                percentile=55,
                smooth_kernel=7,
            )

            # -------------------------------------------------
            # Detect berries
            # -------------------------------------------------

            (
                green_count,
                ripe_count,
                total_berries,
                annotated_bgr,
            ) = predict_berry_count(
                foreground_bgr,
                yolo_model,
                tile_size=640,
                conf=0.4,
            )

            # -------------------------------------------------
            # Predict genotype using DINOv3
            # -------------------------------------------------

            genotype_id, genotype_name = predict_genotype(
                foreground_bgr,
                processor,
                dino_model,
                genotype_classifier,
            )

            # -------------------------------------------------
            # Predict yield
            # -------------------------------------------------

            predicted_yield = predict_yield(
                yield_model=yield_model,
                genotype_id=genotype_id,
                average_berry_weight=average_berry_weight,
                ripe_berry_count=ripe_count,
            )

            # -------------------------------------------------
            # Save results in session state
            # -------------------------------------------------

            st.session_state.prediction_results = {
                "original_image": original_rgb.copy(),
                "foreground_image": cv2.cvtColor(
                    foreground_bgr,
                    cv2.COLOR_BGR2RGB,
                ),
                "annotated_image": cv2.cvtColor(
                    annotated_bgr,
                    cv2.COLOR_BGR2RGB,
                ),
                "green_count": int(green_count),
                "ripe_count": int(ripe_count),
                "total_berries": int(total_berries),
                "genotype_id": genotype_id,
                "genotype_name": genotype_name,
                "average_berry_weight": float(
                    average_berry_weight
                ),
                "predicted_yield": float(predicted_yield),
            }

            # Release large local image arrays after storing
            # the display copies in session state.
            del original_rgb, original_bgr
            del foreground_bgr, annotated_bgr

            gc.collect()

    except Exception as error:
        st.error(
            "The analysis could not be completed."
        )

        st.exception(error)

    finally:
        gc.collect()


# =========================================================
# 18. DISPLAY RESULTS OUTSIDE THE BUTTON BLOCK
# =========================================================
# Results remain visible during normal Streamlit reruns.

results = st.session_state.prediction_results

if results is not None:

    st.divider()
    st.header("Analysis Results")

    # -----------------------------------------------------
    # Images
    # -----------------------------------------------------

    image_col1, image_col2 = st.columns(2)

    with image_col1:
        st.subheader("Original Image")

        st.image(
            results["original_image"],
            use_container_width=True,
        )

    with image_col2:
        st.subheader("Extracted Foreground")

        st.image(
            results["foreground_image"],
            use_container_width=True,
        )

    st.subheader("Detected Berries")

    st.image(
        results["annotated_image"],
        use_container_width=True,
    )

    st.caption(
        "Green boxes indicate green berries, red boxes indicate ripe "
        "berries, and yellow boxes indicate half-ripe berries if that "
        "class is present in the YOLO model."
    )

    # -----------------------------------------------------
    # Berry counts
    # -----------------------------------------------------

    st.divider()
    st.subheader("Berry Detection Summary")

    count_col1, count_col2, count_col3 = st.columns(3)

    with count_col1:
        st.metric(
            "Green Berries",
            results["green_count"],
        )

    with count_col2:
        st.metric(
            "Ripe Berries",
            results["ripe_count"],
        )

    with count_col3:
        st.metric(
            "Total Detected Berries",
            results["total_berries"],
        )

    # -----------------------------------------------------
    # Genotype result
    # -----------------------------------------------------

    st.divider()
    st.subheader("Predicted Blueberry Genotype")

    st.success(
        f"**{results['genotype_name']}**"
    )

    if results["genotype_id"] is not None:
        st.caption(
            f"Genotype class ID: {results['genotype_id']}"
        )

    # -----------------------------------------------------
    # Yield result
    # -----------------------------------------------------

    st.divider()
    st.subheader("Yield Estimation")

    yield_col1, yield_col2 = st.columns(2)

    with yield_col1:
        st.metric(
            "Average Berry Weight",
            f"{results['average_berry_weight']:.2f} g",
        )

    with yield_col2:
        st.metric(
            "Ripe Berry Count Used",
            results["ripe_count"],
        )

    st.metric(
        "Predicted Yield",
        f"{results['predicted_yield']:.2f} g",
    )

    st.caption(
        "The yield prediction uses the predicted genotype, the supplied "
        "average berry weight, and the YOLO-detected ripe berry count. "
        "The result is shown in grams, consistent with the model's "
        "training target."
    )

else:
    st.info(
        "Upload an image, enter the average berry weight, and select "
        "**Analyze Image & Predict Yield** to see the results."
    )
