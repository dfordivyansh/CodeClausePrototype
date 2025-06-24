import streamlit as st
import cv2
from ultralytics import YOLO
from pathlib import Path
import tempfile
import yaml
import matplotlib.pyplot as plt

# --- Paths Setup ---
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
PRED_DIR = BASE_DIR / "predictions_web"
IMG_OUT = PRED_DIR / "images"
LBL_OUT = PRED_DIR / "labels"
YAML_PATH = BASE_DIR / "yolo_params.yaml"

# Create directories
for directory in [PRED_DIR, IMG_OUT, LBL_OUT]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Available Models ---
AVAILABLE_MODELS = {
    "YOLOv8n (Nano)": WEIGHTS_DIR / "yolov8n.pt",
    "YOLOv8s (Small)": WEIGHTS_DIR / "yolov8s.pt",
    "YOLOv8m (Medium)": WEIGHTS_DIR / "yolov8m.pt",
    "YOLOv8l (Large)": WEIGHTS_DIR / "yolov8l.pt",
    "YOLOv8x (X-Large)": WEIGHTS_DIR / "yolov8x.pt",
    "Custom Model": BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
}

# --- Streamlit Page Config ---
st.set_page_config(page_title="🚀 SPACESIGHT", layout="wide")
st.markdown("<h1 style='color:#23297A;'>🚀 SPACESIGHT</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color:#0077B6;'>OBJECT DETECTION IN SPACE STATION ENVIRONMENT</h2>", unsafe_allow_html=True)

# --- Sidebar: Model & Mode Selection ---
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Select YOLO Model", list(AVAILABLE_MODELS.keys()))
model_path = AVAILABLE_MODELS[model_name]
if not model_path.exists():
    st.sidebar.error(f"Model not found: {model_path}")
    st.stop()
model = YOLO(str(model_path))

task = st.sidebar.radio("Select Task", [
    "Batch Upload",
    "Webcam Inference",
    "Test Set Browser",
    "Validation Metrics"
])

# --- Helper: Save Predictions ---
def save_predictions(result, filename):
    # Save annotated image
    img_save = IMG_OUT / filename
    cv2.imwrite(str(img_save), result.plot())
    # Save label txt
    lbl_save = LBL_OUT / Path(filename).with_suffix(".txt");
    with open(lbl_save, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x, y, w, h = box.xywh[0].tolist()
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return img_save, lbl_save

# === Task: Batch Upload ===
if task == "Batch Upload":
    files = st.file_uploader("Upload Images", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if files:
        for f in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
            tmp.write(f.read()); tmp.close()
            results = model.predict(source=tmp.name, conf=0.5)
            res = results[0]
            annotated = res.plot()

            st.image(annotated, caption=f.name, use_column_width=True)
            img_file, lbl_file = save_predictions(res, f.name)

            with open(img_file, 'rb') as img_buf:
                st.download_button("Download Image", img_buf, file_name=img_file.name)
            with open(lbl_file, 'rb') as lbl_buf:
                st.download_button("Download Labels", lbl_buf, file_name=lbl_file.name)

# === Task: Webcam Inference ===
elif task == "Webcam Inference":
    start = st.button("Start")
    stop  = st.button("Stop")
    placeholder = st.empty()
    if start:
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok or stop: break
            res = model.predict(source=frame, conf=0.5)[0]
            annotated = res.plot()
            placeholder.image(annotated, channels="BGR", use_column_width=True)
        cap.release()
        cv2.destroyAllWindows()

# === Task: Test Set Browser ===
elif task == "Test Set Browser":
    if not YAML_PATH.exists():
        st.error("yolo_params.yaml not found.")
    else:
        cfg = yaml.safe_load(open(YAML_PATH))
        img_dir = Path(cfg.get("test", "")) / "images"
        if not img_dir.exists():
            st.error("Test images folder missing.")
        else:
            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            idx = st.slider("Image Index", 0, len(imgs)-1, 0)
            img_path = imgs[idx]
            st.write(img_path.name)
            res = model.predict(source=str(img_path), conf=0.5)[0]
            st.image(res.plot(), use_column_width=True)

# === Task: Validation Metrics ===
elif task == "Validation Metrics":
    if not YAML_PATH.exists():
        st.error("yolo_params.yaml not found.")
    else:
        st.info("Running validation...")
        metrics = model.val(data=str(YAML_PATH), split="test").box
        names = ["Precision", "Recall", "mAP50", "mAP50-95"]
        values = [metrics.p, metrics.r, metrics.map50, metrics.map]
        # Flatten values
        values = [float(v[0]) if hasattr(v, '__len__') else float(v) for v in values]

        col1, col2 = st.columns([1,2])
        with col1:
            for name, val in zip(names, values):
                st.metric(label=name, value=f"{val:.3f}")
        with col2:
            fig, ax = plt.subplots()
            ax.bar(names, values)
            ax.set_ylim(0,1)
            ax.set_ylabel("Score")
            ax.set_title("Validation Metrics")
            st.pyplot(fig)
