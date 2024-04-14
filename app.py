import streamlit as st
from PIL import Image
import os
import cv2
import pytesseract
import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import pandas as pd
import torch
import matplotlib.pyplot as plt

st.title('Hateful Meme Classifier') 
st.text('Project By - Sudheendra,Abhishek,Prakash,Darvesh')

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def preprocess_image(image):
    image = cv2.bilateralFilter(image, 5, 55, 60)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    return image

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    image = preprocess_image(image)
    text = pytesseract.image_to_string(image)
    allowed_chars = string.ascii_letters + string.digits + " "
    filtered_text = "".join(char if char in allowed_chars else " " for char in text).replace("\n", " ")
    return filtered_text

def generate_image_caption(image_path):
    model_name = "t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert(mode="RGB")

    img_str = pytesseract.image_to_string(img)

    inputs = tokenizer("generate a caption for this image: " + img_str, return_tensors="pt").input_ids
    input_ids = inputs.to(device)

    output_ids = model.generate(input_ids, max_length=128)
    preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return preds

def detect_hate_speech(text):
    tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    hate_confidence = predictions[0][1].item()
    classification = "hateful" if hate_confidence > 0.5 else "not hateful"
    confidence = hate_confidence if classification == "hateful" else 1 - hate_confidence
    return classification, confidence

def analyze_meme_for_toxicity(image_path, output_file):
    extracted_text = extract_text_from_image(image_path)
    image_caption = generate_image_caption(image_path)
    combined_text = extracted_text + " " + image_caption

    print("Extracted Text:", extracted_text)
    print("Image Caption:", image_caption)
    print("Combined Text:", combined_text)

    results = {}
    results["extracted_text"] = detect_hate_speech(extracted_text)
    results["image_caption"] = detect_hate_speech(image_caption)
    results["combined_text"] = detect_hate_speech(combined_text)

    with open(output_file, "a") as file:
        file.write(f"Image File Name: {os.path.basename(image_path)}\n")
        for text_type, (classification, confidence) in results.items():
            file.write(f"{text_type.capitalize()} - Classification: {classification}, Confidence: {confidence:.4f}\n")
        file.write("\n")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button('Analyze'):
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        analyze_meme_for_toxicity("temp.jpg", "output.txt")
        
        with open("output.txt", "r") as f:
            results = f.read()
        
        st.text_area("Results", results, height=200)

        
       # Display classification and animated emojis for hateful and not hateful classifications
        if "Combined_text" in results:
            classification_start = results.find("Classification:", results.find("Combined_text")) + len("Classification:")
            classification_end = results.find(",", classification_start)
            classification = results[classification_start:classification_end].strip()
            
            if classification == "hateful":
                st.markdown('<div style="padding: 10px; background-color: #f44336; color: white; border-radius: 5px;">Classification: hateful</div>', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 50px; text-align: center; animation: bounce 1s infinite;">👿</p>', unsafe_allow_html=True)
            elif classification == "not hateful":
                st.markdown('<div style="padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px;">Classification: not hateful</div>', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 50px; text-align: center; animation: heartbeat 1s infinite;">😊</p>', unsafe_allow_html=True)

            

        os.remove("temp.jpg")
        os.remove("output.txt")
