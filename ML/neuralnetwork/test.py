import cv2
import numpy as np
import pytesseract

# sudo apt update
# sudo apt install tesseract-ocr
# sudo apt install libtesseract-dev

# Читать:

# Variant 1
# https://en.wikipedia.org/wiki/Recurrent_neural_network
# https://en.wikipedia.org/wiki/Long_short-term_memory#:~:text=Long%20short%2Dterm%20memory%20(LSTM,networks%2C%20LSTM%20has%20feedback%20connections.
# https://tesseract-ocr.github.io/tessdoc/tess4/NeuralNetsInTesseract4.00.html#:~:text=The%20Tesseract%204.00%20neural%20network,image%20of%20a%20single%20textline.
# https://stackoverflow.com/questions/29560917/does-tessaract-ocr-uses-neural-networks-as-their-default-training-mechanism
# https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/33418.pdf
# https://nanonets.com/blog/ocr-with-tesseract/


# Variant 2
# https://towardsdatascience.com/build-a-multi-digit-detector-with-keras-and-opencv-b97e3cd3b37
# https://colab.research.google.com/drive/1AeQ6_VIrGS6AU8GBTkd8w1wqETy6o5N5#scrollTo=pKZMq9lA3x2P
# http://ufldl.stanford.edu/housenumbers/
# https://www.kaggle.com/code/lifeline/svhn-determine-numbers-from-house-number-plate/data
# https://github.com/thomalm/svhn-multi-digit/blob/master/06-svhn-multi-model.ipynb


if __name__ == '__main__':
    print(pytesseract.get_tesseract_version())

    # Load the img
    # img = cv2.imread("test1.jpg")
    img = cv2.imread("2022.jpg")

    # # Cvt to hsv
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # # Get binary-mask
    # msk = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([179, 255, 255]))
    # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    # dlt = cv2.dilate(msk, krn, iterations=1)
    # thr = 255 - cv2.bitwise_and(dlt, msk)
    #

    # Grayscale, Gaussian blur, Otsu's threshold
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow("o", opening)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    invert = 255 - opening

    # Optical Character Recognition
    d = pytesseract.image_to_string(
        invert,
        lang='eng',
        config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    )
    print(d)
