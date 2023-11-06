import streamlit as st
from PIL import Image, ImageOps,ImageEnhance
import numpy as np
import cv2

# FILTER

# Sepia Filter Function
def apply_sepia_filter(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    return Image.fromarray(sepia_image)

# Blur Filter Function
def apply_blur_filter(image):
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
    return Image.fromarray(blurred_image)

# Invert Filter Function
def apply_invert_filter(image):
    inverted_image = cv2.bitwise_not(image)
    return Image.fromarray(inverted_image)

# Edge Detection (Canny) Filter Function
def apply_edge_detection_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return Image.fromarray(edges, 'L')

# Emboss Filter Function
def apply_emboss_filter(image):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(embossed_image)

# Sharpening Filter Function
def apply_sharpening_filter(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(sharpened_image)

# END FILTER


# Sepia Filter Function
def apply_sepia_filter(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    return Image.fromarray(sepia_image)

# Blur Filter Function
def apply_blur_filter(image):
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
    return Image.fromarray(blurred_image)

# Invert Filter Function
def apply_invert_filter(image):
    inverted_image = cv2.bitwise_not(image)
    return Image.fromarray(inverted_image)

# Edge Detection (Canny) Filter Function
def apply_edge_detection_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return Image.fromarray(edges, 'L')

# Emboss Filter Function
def apply_emboss_filter(image):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(embossed_image)

# Sharpening Filter Function
def apply_sharpening_filter(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(sharpened_image)

# END FILTER

# page configurations
st.set_page_config(page_title="PixelPerfect ")

st.markdown(
    f"""
    <div style="background-color: #ff8000; padding: 10px; border-radius: 5px; text-align: center;">
        <h1 style="color: white;">PixelPerfect</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.header("Your Ultimate Image Enhancement Tool ")
st.subheader("Upload an image to get started 🚀")




# Display the Streamlit app
st.write("""
    Instructions:
    1. Upload images using the "Upload Images" section.
    2. Select an image from the collection.
    3. Choose a filter to apply from the "Apply Filters" section.
    """)

# Upload an image
uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    try:
        # Use PIL to open the uploaded image
        image = Image.open(uploaded_image)

        if image is not None:
            # Display the original image
            left_column, right_column = st.columns(2)
            with left_column:
                st.header("Original Image")
                st.image(image, use_column_width=True, caption='Original Image')
            with right_column:
                st.header("Output Image")
            # writing settings code
            st.sidebar.write("Settings")
            setting_sharp = st.sidebar.slider("Sharpness", 0.0, 3.0, step=0.5)
            setting_color = st.sidebar.slider("Color Intensity", 0.0, 3.0, step=0.5)
            setting_brightness = st.sidebar.slider("Brightness", 0.0, 3.0, step=0.5)
            setting_contrast = st.sidebar.slider("Contrast", 0.0, 3.0, step=0.5)

            setting_flip_image = st.sidebar.selectbox("Flip Image", options=(
                "select flip direction", "FLIP_TOP_BOTTOM", "FLIP_LEFT_RIGHT","ROTATE_90","ROTATE_180","ROTATE_270"))

            # FILTER OPTIONS
            st.sidebar.write("Filters")
            fiiter_Grayscale = st.sidebar.checkbox("Grayscale")
            filter_Sepia = st.sidebar.checkbox("Sepia")
            filter_blur = st.sidebar.checkbox("Blur")
            filter_Invert = st.sidebar.checkbox("Invert")
            filter_Edge_Detection = st.sidebar.checkbox("Edge Detection")
            filter_Emboss = st.sidebar.checkbox("Emboss")
            filter_Sharpen = st.sidebar.checkbox("Sharpening")

            # adding grain effect to the sidebar
            st.sidebar.write("Grain Effect")
            grain_intensity = st.sidebar.slider("Intensity", 0, 100, 0)

            # checking setting_sharp value
            if setting_sharp:
                sharp_value = setting_sharp
            else:
                sharp_value = 0

            # checking color
            if setting_color:
                set_color = setting_color
            else:
                set_color = 1

            # checking brightness
            if setting_brightness:
                set_brightness = setting_brightness
            else:
                set_brightness = 1

            # checking contrast
            if setting_contrast:
                set_contrast = setting_contrast
            else:
                set_contrast = 1    
            
            # checking setting_flip_image
            flip_direction = setting_flip_image

            # implementing sharpness
            sharp = ImageEnhance.Sharpness(image)
            edited_img = sharp.enhance(sharp_value)

            # implementing colors
            color = ImageEnhance.Color(edited_img)
            edited_img = color.enhance(set_color)

            # implementing brightness
            brightness = ImageEnhance.Brightness(edited_img)
            edited_img = brightness.enhance(set_brightness)

            # implementing contrast
            contrast = ImageEnhance.Contrast(edited_img)
            edited_img = contrast.enhance(set_contrast)

            # implementing flip direction
            if flip_direction == "FLIP_TOP_BOTTOM":
                edited_img = edited_img.transpose(Image.FLIP_TOP_BOTTOM)
            elif flip_direction == "FLIP_LEFT_RIGHT":
                edited_img = edited_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_direction == "ROTATE_90":
                edited_img = edited_img.transpose(Image.ROTATE_90)
            elif flip_direction == "ROTATE_180":
                edited_img = edited_img.transpose(Image.ROTATE_180)
            elif flip_direction == "ROTATE_270":
                edited_img = edited_img.transpose(Image.ROTATE_270)
            else:
                pass

            # implementing filters
            if fiiter_Grayscale:
                edited_img = edited_img.convert("L")
            if filter_Sepia:
                edited_img = apply_sepia_filter(np.array(edited_img))

            if filter_blur:
                edited_img = apply_blur_filter(np.array(edited_img))

            if filter_Invert:
                edited_img = apply_invert_filter(np.array(edited_img))

            if filter_Edge_Detection:
                edited_img = apply_edge_detection_filter(np.array(edited_img))

            if filter_Emboss:
                edited_img = apply_emboss_filter(np.array(edited_img))

            if filter_Sharpen:
                edited_img = apply_sharpening_filter(np.array(edited_img))
            # Check if grain effect is applied
            if grain_intensity > 0:
                img_array = np.array(edited_img)
                height, width, a = img_array.shape

                # Generate random noise
                noise = np.random.randint(-grain_intensity, grain_intensity, (height, width, 3))

                # Apply noise to the image
                noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

                edited_img = Image.fromarray(noisy_img_array)
            st.sidebar.write("ADD TEXT")
            text = st.sidebar.text_input("Text to overlay on the image:")
            wi,hi = edited_img.size
            setting_width  = st.sidebar.slider("Width", 0, wi, step=1)
            setting_height  = st.sidebar.slider("height", 0, hi, step=1)
            setting_font_scale  = st.sidebar.slider("Font Scale", 0, 10, step=1)
            setting_font_a  = st.sidebar.slider("RED", 0, 255, step=1)
            setting_font_b  = st.sidebar.slider("GREEN", 0, 255, step=1)
            setting_font_c  = st.sidebar.slider("BLUE", 0, 255, step=1)
            # Calculate the position to place the text (you can adjust this)
            if setting_width:
                text_x = setting_width
            else:
                text_x = 0

            if setting_width:
                text_y = setting_height
            else:
                text_y = 0

            if setting_font_scale:
                set_font_scale = setting_font_scale
            else:
                set_font_scale = 0
            
            if setting_font_a:
                set_font_a = setting_font_a
            else:
                set_font_a = 0
            
            if setting_font_b:
                set_font_b = setting_font_b
            else:
                set_font_b = 0

            if setting_font_c:
                set_font_c = setting_font_c
            else:
                set_font_c = 0
            

            if st.sidebar.button("Add Text"):
                # Convert Image to NumPy array
                image_np = np.array(edited_img)
                # Define the font and other text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = set_font_scale
                font_color = (set_font_a, set_font_b, set_font_c)  
                font_thickness = 2

                

                # Use OpenCV to add the text overlay to the image
                cv2.putText(image_np, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                # Convert the NumPy array back to an Image for display
                edited_img = Image.fromarray(image_np)
            with right_column:
                st.image(edited_img, width=400)

            st.write(">To download edited image right click and click save image as.")
        else:
            st.warning('Invalid image format. Please upload a valid image file.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

# Create a Streamlit column layout to mimic the CSS code
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <style>
        .footer {
          position: fixed;
          left: 0;
          bottom: 0;
          width: 100%;
          background-color: #445c5a;
          color: white;
          text-align: center;
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="footer">
            <h3 style="text-transform: uppercase;">Developers</h3>
            <ul style="list-style-type: none; padding: 0;">
                <li>
                    <a href="/saurabh" style="color: white; text-decoration: none;">Saurabh Mulik</a>
                </li>
                <li>
                    <a href="/yuvraj" style="color: white; text-decoration: none;">Yuvraj Patare</a>
                </li>
                <li>
                    <a href="/sagar" style="color: white; text-decoration: none;">Sagar Kengar</a>
                </li>
                <li>
                    <a href="/rupesh" style="color: white; text-decoration: none;">Rupesh Pingale</a>
                </li>
            </ul>
            <divstyle => &copy; 2023 PixelPerfect . All rights reserved. </div>
        </div>
        """,
        unsafe_allow_html=True
    )