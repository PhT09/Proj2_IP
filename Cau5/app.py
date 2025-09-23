import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from supporting_functions import *

from functions.edge_detection import *
from functions.noise_reduce import *
from functions.sharpening import *

FUNCTIONS = {
    "Làm mờ": {
        "Gaussian": ("GaussianNoiseReduction", ["kernel_size", "sigma"]),
        "Median": ("MedianNoiseReduction", ["kernel_size"]),
        "Average": ("BlurNoiseReduction", ["kernel_size"]),
        "Bilateral": ("BilateralNoiseReduction", ["diameter", "sigma_color", "sigma_space"]),
    },
    "Làm sắc nét": {
        "Unsharp Masking": ("UnsharpMasking", ["kernel_size", "sigma", "alpha"]),
        "Laplacian": ("LaplacianSharpening", ["kernel_size", "alpha"]),
    },
    "Phát hiện cạnh": {
        "Sobel": ("SobelEdgeDetection", ["kernel_size", "threshold1", "threshold2"]),
        "Canny": ("CannyEdgeDetection", ["threshold1", "threshold2"]),
        "Laplacian": ("LaplacianEdgeDetection", ["kernel_size"]),
    }
}


st.set_page_config(page_title="Ứng dụng xử lý ảnh", layout="wide")

st.markdown(
    """
    <style>
    /* Giảm khoảng cách trên cùng */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(['Xử lý ảnh', 'Phân tích'])

uploaded_file, result, main_function = None, None, None

with tab1:
    col_left, col_right = st.columns([1,2]) 

    with col_left:
        st.header("Điều khiển")
        uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_array = load_image(uploaded_file)

            # Chọn chức năng
            main_function = st.selectbox("Chọn chức năng", list(FUNCTIONS.keys()))
            method = st.selectbox("Chọn phương pháp", list(FUNCTIONS[main_function].keys()))

            # Lấy hàm và tham số cần thiết
            func_name, params = FUNCTIONS[main_function][method]
            func = globals()[func_name]   # gọi hàm theo tên

            # Tạo slider input cho từng tham số
            kwargs = {}
            for p in params:
                if p == "kernel_size":
                    kwargs[p] = st.slider("Kernel size", 1, 11, 3, step=2)
                elif p == "sigma":
                    kwargs[p] = st.slider("Sigma", 1, 10, 2)
                elif p == "alpha":
                    kwargs[p] = st.slider("Alpha", 1.0, 3.0, 1.5)
                elif p == "diameter":
                    kwargs[p] = st.slider("Diameter", 1, 15, 9)
                elif p == "sigma_color":
                    kwargs[p] = st.slider("Sigma Color", 1, 150, 75)
                elif p == "sigma_space":
                    kwargs[p] = st.slider("Sigma Space", 1, 150, 75)
                elif p == "threshold1":
                    kwargs[p] = st.slider("Ngưỡng dưới", 20, 200, 100)
                elif p == "threshold2":
                    kwargs[p] = st.slider("Ngưỡng trên", 80, 250, 200)

            # Gọi hàm xử lý ảnh
            result = func(img_array, **kwargs)

            if main_function == "Làm mờ":
                psnr_val = psnr(img_array, result)
                ssim_val = ssim(img_array, result)

                st.markdown(f"**PSNR:** {psnr_val:.2f} dB")
                st.markdown(f"**SSIM:** {ssim_val:.4f}")

            # Encode ảnh sang PNG 
            is_success, buffer = cv2.imencode(".png", result)
            byte_im = buffer.tobytes()

            st.download_button(
                label="Tải ảnh kết quả",
                data=byte_im,
                file_name="result.png",
                mime="image/png"
            )
        else:
            st.info("Hãy tải ảnh để bắt đầu.")

    with col_right:
        st.header('Hiển thị ảnh')

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(resize_for_display(img_array), caption="Ảnh gốc", use_container_width=False)
            with col2:
                st.image(resize_for_display(result), caption="Ảnh kết quả", use_container_width=False)
        else:
            st.info("Hãy tải ảnh để bắt đầu.")

with tab2:
    if uploaded_file is not None and result is not None:
        # ---- 1. Overlay biên ----
        if main_function == "Phát hiện cạnh":
            st.subheader("Chồng biên lên ảnh gốc")
            edges_overlay = overlay_edges(img_array, result, color=(255, 0, 0))
            st.image(edges_overlay, caption="Ảnh gốc với biên màu đỏ")

        # ---- 2. So sánh hiệu suất ----
        if main_function == "Làm mờ":
            st.subheader("So sánh hiệu suất các phương pháp")
            # Tính PSNR, SSIM của nhiều phương pháp
            methods = {
                "Gaussian": GaussianNoiseReduction(img_array, 5, 1),
                "Median": MedianNoiseReduction(img_array, 5),
                "Bilateral": BilateralNoiseReduction(img_array, 9, 75, 75),
            }
            metrics = []
            for name, res in methods.items():
                psnr_val = psnr(img_array, res)
                ssim_val = ssim(img_array, res, channel_axis=-1)
                metrics.append((name, psnr_val, ssim_val))

            # Hiển thị bảng kết quả
            import pandas as pd
            df = pd.DataFrame(metrics, columns=["Phương pháp", "PSNR (dB)", "SSIM"])
            df.index = [i+1 for i in range(len(metrics))]
            st.table(df)

            # Hiển thị ảnh để so sánh
            cols = st.columns(len(methods))
            for i, (name, res) in enumerate(methods.items()):
                with cols[i]:
                    st.image(res, caption=name, use_container_width=True)

        if main_function == "Làm sắc nét":
            st.subheader("So sánh hiệu suất các phương pháp")
            # Tính PSNR, SSIM của nhiều phương pháp
            methods = {
                "Laplacian": LaplacianSharpening(img_array, 5, 1.25),
                "Unsharp Masking": UnsharpMasking(img_array, 5, 2, 1.25)
            }
            metrics = []
            for name, res in methods.items():
                psnr_val = psnr(img_array, res)
                ssim_val = ssim(img_array, res, channel_axis=-1)
                metrics.append((name, psnr_val, ssim_val))

            # Hiển thị bảng kết quả
            import pandas as pd
            df = pd.DataFrame(metrics, columns=["Phương pháp", "PSNR (dB)", "SSIM"])
            df.index = [i+1 for i in range(len(metrics))]
            st.table(df)

            # Hiển thị ảnh để so sánh
            cols = st.columns(len(methods))
            for i, (name, res) in enumerate(methods.items()):
                with cols[i]:
                    st.image(res, caption=name, use_container_width=True)

    else:
        st.info("Hãy thực hiện xử lý ảnh trước.")
    