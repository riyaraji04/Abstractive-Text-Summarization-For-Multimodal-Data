#!/usr/bin/env python

from PIL import Image
import pytesseract
import sys

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
im = Image.open("uploads/uploadedimage.jpg")
text = pytesseract.image_to_string(im,lang="eng")
file = open("uploads/uploadedtext.txt", "w")
file.write(text)
