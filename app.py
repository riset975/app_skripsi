import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Gunakan urutan kelas sesuai yang dipakai saat training (dari Colab/Roboflow)
class_names = ["objects", "Cardiomegaly", "Nodule-Mass", "Pneumothorax"]

@st.cache_resource
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file("config.yaml")
    cfg.MODEL.WEIGHTS = "best_model.pth"
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3

    # Metadata harus sama seperti saat training
    metadata = MetadataCatalog.get("roboflow_train")
    metadata.set(thing_classes=class_names)

    predictor = DefaultPredictor(cfg)
    return predictor, metadata

predictor, metadata = load_model()

st.title("Aplikasi Deteksi Abnormalitas Paru-Paru Pada Citra X-Ray Dada")
st.write("Aplikasi ini mendeteksi secara otomatis kelainan pada paru-paru yaitu Cardiomegaly, Nodule/Mass, dan Pneumothorax pada citra X-ray dada.")

uploaded_file = st.file_uploader("Upload Citra X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)[:, :, ::-1]  # Convert RGB (PIL) to BGR (OpenCV)

    st.image(image, caption="Gambar Input", use_column_width=True)

    outputs = predictor(img)

    # Visualisasi
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = out.get_image()[:, :, ::-1]

    st.image(result_img, caption="Hasil Prediksi", use_column_width=True)

    # Tampilkan nama kelas dan confidence
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    st.write("### Deteksi:")
    for cls, score in zip(pred_classes, scores):
        try:
            class_name = class_names[cls]
        except IndexError:
            class_name = f"Unknown({cls})"
        st.write(f"- {class_name} dengan confidence {score:.2f}")

else:
    st.info("Silakan Upload Citra X-Ray untuk melihat hasil prediksi.")
