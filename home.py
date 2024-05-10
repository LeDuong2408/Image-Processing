import streamlit as st
import cv2
import numpy as np
from PIL import Image
import Chapter03 as c3
import Chapter04 as c4
import Chapter05 as c5
import Chapter09 as c9

def main():
    options_1 = ['Open', 'OpenColor', 'Save', 'Exit']
    options_2 = ['Chương 3', 'Chương 4', 'Chương 5', 'Chương 9']

    if 'options_3' not in st.session_state:
        st.session_state.options_3 = ['Negative', 'Logarit', 'PiecewiseLinear', 'Histogram', 'HistEqual', 'HistEqualColor',
                 'LocalHist', 'HistStat', 'BoxFilter', 'LowpassGauss', 'Threshold', 'MedianFilter', 'Sharpen']
        print ('Chua load')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button('Chương 3'):
            st.session_state.options_3 = ['Negative', 'Logarit', 'PiecewiseLinear', 'Histogram', 'HistEqual', 'HistEqualColor',
                 'LocalHist', 'HistStat', 'BoxFilter', 'LowpassGauss', 'Threshold', 'MedianFilter', 'Sharpen', 'Gradient']
    with col2:
        if st.button('Chương 4'):
            st.session_state.options_3 = ['Spectrum', 'FrequencyFilter', 'RemoveMoire']
    with col3:
        if st.button('Chương 5'):
            st.session_state.options_3 = ['CreateMotionNoise', 'DenoiseMotion', 'DenoisestMotion']
    with col4:
        if st.button('Chương 9'):
            st.session_state.options_3 = ['ConnectedComponent', 'CountRice']

    selectedBox = st.selectbox('Chọn phương pháp xử lí ảnh:', st.session_state.options_3)   

    image_file = st.file_uploader("Upload Images", type=["bmp","png","jpg","jpeg", "tif"])
    def onOpen():
        if image_file is not None:
            st.session_state.imageIn = Image.open(image_file)
    def onNegative():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.Negative(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onLogarit():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.Logarit(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onPiecewiseLinear():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.PiecewiseLinear(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onHistogram():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.Histogram(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onHistEqual():
        img_array = np.array(st.session_state.imageIn)
        imgout = cv2.equalizeHist(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onHistEqualColor():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.HistEqualColor(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onLocalHist():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.LocalHist(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onHistStat():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.HistStat(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onBoxFilter():
        img_array = np.array(st.session_state.imageIn)
        imgout = cv2.blur(img_array,(21,21))
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onLowpassGauss():
        img_array = np.array(st.session_state.imageIn)
        imgout = cv2.GaussianBlur(img_array,(43,43),7.0)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onThreshold():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.Threshold(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onMedianFilter():
        img_array = np.array(st.session_state.imageIn)
        imgout = cv2.medianBlur(img_array, 7)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onSharpen():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.Sharpen(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onGradient():
        img_array = np.array(st.session_state.imageIn)
        imgout = c3.Gradient(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onSpectrum():
        img_array = np.array(st.session_state.imageIn)
        imgout = c4.Spectrum(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onFrequencyFilter():
        img_array = np.array(st.session_state.imageIn)
        imgout = c4.FrequencyFilter(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onDrawNotchRejectFilter():
        imgout = c4.DrawNotchRejectFilter()
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onRemoveMoire():
        img_array = np.array(st.session_state.imageIn)
        imgout = c4.RemoveMoire(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    def onCreateMotionNoise():
        img_array = np.array(st.session_state.imageIn)
        imgout = c5.CreateMotionNoise(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onDenoiseMotion():
        img_array = np.array(st.session_state.imageIn)
        imgout = c5.DenoiseMotion(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onDenoisestMotion():
        img_array = np.array(st.session_state.imageIn)
        temp = cv2.medianBlur(img_array, 7)
        imgout = c5.DenoiseMotion(temp)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onErosion():
        img_array = np.array(st.session_state.imageIn)
        imgout = np.array(st.session_state.imageIn)
        c9.Erosion(img_array, imgout)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onDilation():
        img_array = np.array(st.session_state.imageIn)
        imgout = np.array(st.session_state.imageIn)
        c9.Dilation(img_array, imgout)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onOpeningClosing():
        img_array = np.array(st.session_state.imageIn)
        imgout = np.array(st.session_state.imageIn)
        c9.OpeningClosing(img_array, imgout)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onBoundary():
        img_array = np.array(st.session_state.imageIn)
        imgout = c9.Boundary(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onConnectedComponent():
        img_array = np.array(st.session_state.imageIn)
        imgout = c9.ConnectedComponent(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))

    def onCountRice():
        img_array = np.array(st.session_state.imageIn)
        imgout = c9.CountRice(img_array)
        st.session_state.imageOut = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    if (st.button('Open')):
        onOpen()
    if 'imageIn' in st.session_state:
        if (selectedBox == "Negative"):
            onNegative()
    if "imageIn" in st.session_state:
        st.image(st.session_state.imageIn)
        if (selectedBox == "Negative"):
            onNegative()
        elif (selectedBox == "Logarit"):
            onLogarit()
        elif (selectedBox == "PiecewiseLinear"):
            onPiecewiseLinear()
        elif (selectedBox == "Histogram"):
            onHistogram()
        elif (selectedBox == "HistEqual"):
            onHistEqual()
        elif (selectedBox == "HistEqualColor"):
            onHistEqualColor()
        elif (selectedBox == "LocalHist"):
            onLocalHist()
        elif (selectedBox == "HistStat"):
            onHistStat()
        elif (selectedBox == "BoxFilter"):
            onBoxFilter()
        elif (selectedBox == "LowpassGauss"):
            onLowpassGauss()
        elif (selectedBox == "Threshold"):
            onThreshold()
        elif (selectedBox == "MedianFilter"):
            onMedianFilter()
        elif (selectedBox == "Sharpen"):
            onSharpen()
        elif (selectedBox == "Gradient"):
            onGradient()
        elif (selectedBox == "Spectrum"):
            onSpectrum()
        elif (selectedBox == "FrequencyFilter"):
            onFrequencyFilter()
        elif (selectedBox == "DrawNotchRejectFilter"):
            onDrawNotchRejectFilter()
        elif (selectedBox == "RemoveMoire"):
            onRemoveMoire()
        elif (selectedBox == "CreateMotionNoise"):
            onCreateMotionNoise()
        elif (selectedBox == "DenoiseMotion"):
            onDenoiseMotion()
        elif (selectedBox == "DenoisestMotion"):
            onDenoisestMotion()
        elif (selectedBox == "Erosion"):
            onErosion()
        elif (selectedBox == "Dilation"):
            onDilation()
        elif (selectedBox == "OpeningClosing"):
            onOpeningClosing()
        elif (selectedBox == "Boundary"):
            onBoundary()
        elif (selectedBox == "ConnectedComponent"):
            onConnectedComponent()
        elif (selectedBox == "CountRice"):
            onCountRice()
            
        if (st.button('Xử lý')):
            if "imageOut" in st.session_state:
                st.image(st.session_state.imageOut)