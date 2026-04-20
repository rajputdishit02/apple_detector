import os
import tempfile
from collections import defaultdict
from xml.parsers.expat import model

import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.transforms import functional as F

from train_rcnn import get_model_instance_segmentation


MODEL_PATH = r"C:\Dishit Rajput\Projects\apple_detector\MinneApple\model_0.pth"
DEVICE = "cpu"
NUM_CLASSES = 2  # background + apple


@st.cache_resource
def load_model():
    model = get_model_instance_segmentation(NUM_CLASSES, "frcnn")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_bgr, confidence_threshold):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    results = []
    for box, score in zip(boxes, scores):
        if float(score) >= confidence_threshold:
            x1, y1, x2, y2 = box.astype(int)
            results.append((x1, y1, x2, y2, float(score)))

    return results


def draw_boxes(image_bgr, detections):
    output = image_bgr.copy()

    for x1, y1, x2, y2, score in detections:
        # Calculate center and radius for the circle
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max((x2 - x1) // 2, (y2 - y1) // 2)
        
        # Draw the circle
        cv2.circle(output, center, radius, (0, 255, 0), 2)

        label = f"apple {score:.2f}"
        # Place label above the circle
        text_y = center[1] - radius - 10
        if text_y < 20:
            text_y = center[1] + radius + 20

        cv2.putText(
            output,
            label,
            (center[0] - 50, text_y),  # Center the text horizontally on the circle
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return output


st.title("Apple Detection App")
st.write("Upload one or more apple tree images and detect apples with bounding boxes.")

confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

uploaded_files = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    model = load_model()

    for uploaded_file in uploaded_files:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.warning(f"Could not read file: {uploaded_file.name}")
            continue

        detections = predict_image(model, image_bgr, confidence_threshold)
        boxed_image = draw_boxes(image_bgr, detections)

        boxed_image_rgb = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)

        st.subheader(uploaded_file.name)
        st.write(f"Detected apples: {len(detections)}")
        st.image(boxed_image_rgb, caption=uploaded_file.name, use_container_width=True)