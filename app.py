from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import streamlit as st
import numpy as np

model = YOLO('model/YOLOv8_VR_Glasses.pt')
title = st.text_input('Insert Image Link')

if title:
  result = model(
    source = title,
    save=False)

  for r in result:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

  st.image(im)
