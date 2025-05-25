import streamlit as st
from PIL import Image
from captioning import generate_caption
from segmentation import segment_image

st.set_page_config(page_title="Image Captioning + Segmentation", layout="centered")
st.title("🧠 Image Captioning + Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
        st.success("Caption generated!")
        st.markdown(f"**Caption:** {caption}")

    with st.spinner("Segmenting image..."):
        segmented = segment_image(image)
        st.success("Segmentation complete!")
        st.image(segmented, caption="Segmented Output", use_column_width=True)

