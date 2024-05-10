import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
import joblib
import math
from Giai_PT_Bac_2_Streamlit import giai_pt_bac_2 as ptb2
from NhanDangKhuonMat_onnx_Streamlit import predict as ndkm
from NhanDangTraiCay_Onnx_Streamlit import nhan_dang_trai_cay as ndtc
from nhan_dang_chu_so_mnist_streamlit import home as ndcs
from Phat_Hien_Doi_Tuong_Yolo4_streamlit import home as phdt
import home as hm

def main(): 
    st.title("Streamlit App")

    # Sidebar
    st.sidebar.header("Options")
    selected_option = st.sidebar.selectbox("Select an option", ["Home","Bài tập", "Nhận dạng khuôn mặt", "Giải phương trình bậc 2", "Nhận diện trái cây", "Nhận dạng chữ số viết tay", "Nhận dạng đối tượng", "About"])

    if selected_option == "Home":
        show_home()
    elif selected_option == "Nhận dạng khuôn mặt":
        ndkm.run_face_recognition_app()
    elif selected_option == "Giải phương trình bậc 2":
        ptb2.run_giai_ptb2()
    elif selected_option == "Nhận diện trái cây":
        ndtc.main()
    elif selected_option == "Nhận dạng chữ số viết tay":
        ndcs.main()
    elif selected_option == "Nhận dạng đối tượng":
        phdt.main()
    elif selected_option == "About":
        show_about()
    elif selected_option == "Bài tập":
        hm.main()

def show_home():
    st.header("Name: Lê Dưỡng")
    st.header("Student ID: 21110412")
    st.header("Bài báo cáo cuối kì môn Xử lý ảnh số")
    st.header("Giáo viên hướng dẫn: TS.Trần Tiến Đức")



def show_about():
    st.header("About")
    st.write("Đây là sản phẩm báo cáo cuối kỳ của Lê Dưỡng môn Xử lý ảnh số")

if __name__ == "__main__":
    main()
