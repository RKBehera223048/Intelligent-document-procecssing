import kagglehub

path = kagglehub.dataset_download("urbikn/sroie-datasetv2")

print("Path to dataset files:", path)

! cat /kaggle/input/sroie-datasetv2/SROIE2019/test/entities/X00016469670.txt

! pip install opencv-python matplotlib numpy

import os

one_image_path = os.path.join(path, 'SROIE2019', 'train', 'img', 'X51005453729.jpg')

import matplotlib.pyplot as plt

import cv2

import numpy as np

def display_image(image, title="Image"):

    plt.figure(figsize=(7, 7))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.title(title)

    plt.axis('off')

    plt.show()

one_image = cv2.imread(one_image_path)

display_image(one_image, "Original Receipt Image")

def convert_to_grayscale(image):

  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grayscale_image = convert_to_grayscale(one_image)

display_image(grayscale_image, "Grayscale Image")

def reduce_noise(gray_image):

  return cv2.GaussianBlur(gray_image, (5, 5), 0)

blur_reduced_image = reduce_noise(grayscale_image)

display_image(blur_reduced_image, "Blur Reduced Image")

def binarize_image(blur_reduced_image):

  return cv2.adaptiveThreshold(

    blur_reduced_image,

    255,

    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

    cv2.THRESH_BINARY,

    11,

    4

  )

binarized_image = binarize_image(blur_reduced_image)

display_image(binarized_image, "Binarized Image")

def deskew_image(image):

    coords = cv2.findNonZero(image)

    rect = cv2.minAreaRect(coords)

    angle = rect[-1] - 90

    if angle < -45:

        angle = -(90 + angle)

    else:

        angle = angle

    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h),

                             flags=cv2.INTER_CUBIC,

                             borderMode=cv2.BORDER_REPLICATE)

    print(f"Detected skew angle: {angle:.2f} degrees")

    (h, w) = rotated.shape

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    deskewed_gray = cv2.warpAffine(rotated, M, (w, h),

                                  flags=cv2.INTER_CUBIC,

                                  borderMode=cv2.BORDER_REPLICATE)

    return deskewed_gray

deskewed_image = deskew_image(binarized_image)

display_image(deskewed_image, "Deskewed Image")

def process_one_image(image):

  image = convert_to_grayscale(image)

  print("Converted image to grayscale..")

  image = reduce_noise(image)

  print("Reduced noise in the image..")

  image = binarize_image(image)

  print("Binarized the image..")

  image = deskew_image(image)

  print("Corrected image orientation..")

  return image

import time

output_folder_path = "/content/processed_images"

start_time = time.time()

if os.makedirs(output_folder_path, exist_ok=True):

  print(f"Created folder: {output_folder_path}")

for image_name in os.listdir(os.path.join(path, 'SROIE2019', 'train', 'img'))[:20]:

  print(f"Processing image: {image_name}")

  image_path = os.path.join(path, 'SROIE2019', 'train', 'img', image_name)

  image = cv2.imread(image_path)

  processed_image = process_one_image(image)

  output_path = os.path.join(output_folder_path, image_name)

  cv2.imwrite(output_path, processed_image)

  print(f"Saved processed image to: {output_path}")

  print("-"*50)

print("Processing images is completed.")

print(f"Total time taken: {time.time() - start_time} seconds")

! pip install pytesseract pillow

from PIL import Image

import pytesseract

pytesseract.image_to_string(Image.open('/content/processed_images/X51006414392.jpg'))

from PIL import Image

import pytesseract

import time

input_folder_path = "/content/processed_images"

output_folder_path = "/content/tesseract_output"

start_time = time.time()

if os.makedirs(output_folder_path, exist_ok=True):

  print(f"Created folder: {output_folder_path}")

total_images = sum(1 for entry in os.scandir(input_folder_path))

print(f"Total images in folder: {total_images}")

for i, image_name in enumerate(os.listdir(input_folder_path)[:20], 1):

  print(f"Processing image {i}/{total_images}: {image_name}")

  image_path = os.path.join(input_folder_path, image_name)

  print("Extracting text from image..")

  text = pytesseract.image_to_string(Image.open(image_path))

  output_path = os.path.join(output_folder_path, image_name.replace(".jpg", ".txt"))

  with open(output_path, "w") as f:

    f.write(text)

  print(f"Saved extracted text to {output_path}")

  print("-"*50)

print("Text Extraction Completed.")

print(f"Total time taken: {time.time() - start_time} seconds")

prompt = """
Extract the information from the given image.
Information to be extracted: company, date, address, total.
The image has been converted to grayscale, noise reduced, binarized, and deskewed using opencv.
Always give your response in the following format:
{
    "company": "COMPANY_NAME",
    "date": "DATE",
    "address": "ADDRESS",
    "total": "TOTAL",
}
Also, the text has been extracted from the image using tesseract.
Use the extracted text as support for extracting information.
If you believe the text extraction is incorrect somewhere, you may correct it yourself and provide corrected information.
Respond with the extracted information only in the specified format.
Here is the text:

"""

from google import genai

from google.colab import userdata

from PIL import Image

import json

import time

genai_client = genai.Client(api_key=userdata.get('GOOGLE_API_KEY'))

image_folder_path = "/content/processed_images"

text_folder_path = "/content/tesseract_output"

output_folder_path = "/content/json_output"

start_time = time.time()

if os.makedirs(output_folder_path, exist_ok=True):

  print(f"Created folder: {output_folder_path}")

total_images = sum(1 for entry in os.scandir(image_folder_path))

print(f"Total images in folder: {total_images}")

for i, image_name in enumerate(os.listdir(input_folder_path)[:20], 1):

  print(f"Processing image {i}/{total_images}: {image_name}")

  image_path = os.path.join(input_folder_path, image_name)

  print(f"Loading image: {image_path}")

  with open(image_path, "rb") as f:

    image = Image.open(image_path)

  text_path = os.path.join(text_folder_path, image_name.replace(".jpg", ".txt"))

  print(f"Loading extracted text: {text_path}")

  with open(text_path, "r") as f:

    text = f.read()

  print("Extracting information from image and text..")

  prompt = prompt + text

  contents = [

        image,

        {

            "text": prompt

        }

    ]

  response = genai_client.models.generate_content(model='gemini-2.5-flash', contents=contents)

  usage_metadata = response.usage_metadata

  print(f"Input Token Count: {usage_metadata.prompt_token_count}")

  print(f"Thoughts Token Count: {response.usage_metadata.thoughts_token_count}")

  print(f"Output Token Count: {usage_metadata.candidates_token_count}")

  print(f"Total Token Count: {usage_metadata.total_token_count}")

  extracted_information = json.loads(response.text.replace('```json', '').replace('```', ''))

  output_path = os.path.join(output_folder_path, image_name.replace(".jpg", ".json"))

  with open(output_path, "w") as f:

    json.dump(extracted_information, f, indent=4)

  print(f"Saved extracted information to {output_path}")

  print("-"*50)

  time.sleep(60)

print("Information Extraction Completed.")

print(f"Total time taken: {time.time() - start_time} seconds")
