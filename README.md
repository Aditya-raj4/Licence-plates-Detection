# Project: Automatic License Plate Recognition (ALPR) System

Description:
This project aimed to develop an Automatic License Plate Recognition (ALPR) system using Python and OpenCV. The goal was to build a system that could detect and extract license plates from images or video feeds and recognize the characters on the plate. The system was designed to automate vehicle identification tasks in parking management, toll booths, and security systems.

Tasks and Responsibilities:
Image Acquisition and Preprocessing:

Collected a dataset of vehicle images containing visible license plates.
Performed image preprocessing using OpenCV:
Converted images to grayscale to reduce complexity.
Applied Gaussian Blur to reduce noise and improve edge detection.
Used histogram equalization to enhance image contrast for better visibility of license plates.
License Plate Detection:

Used edge detection techniques like the Canny edge detector to highlight the contours of the license plate.
Applied contour detection to find the rectangular contours that resembled the shape of a license plate.
Filtered contours based on aspect ratio, size, and geometric properties to isolate the license plate region.
Character Segmentation:

Once the license plate was detected, it was extracted from the image using image cropping.
Converted the extracted plate image to binary (black and white) using adaptive thresholding.
Applied morphological operations (e.g., dilation and erosion) to clean up noise and separate the characters.
Used connected component analysis to segment individual characters from the license plate.
Optical Character Recognition (OCR):

Integrated the Tesseract OCR engine with OpenCV to recognize the segmented characters from the license plate.
Preprocessed the character images further to optimize them for OCR, including resizing and normalizing.
Extracted the license plate number and formatted it for output.
System Deployment:

Implemented the system to process both images and real-time video feeds from a camera.
The real-time system used OpenCV to continuously capture frames, detect license plates, and recognize characters.
Displayed the detected license plate number on the video feed in real time.
Technologies Used:
Python: For scripting and implementation.
OpenCV: For all image processing tasks, including edge detection, contour detection, and morphological transformations.
Tesseract OCR: For character recognition from the segmented license plate.
NumPy: For matrix operations and image manipulation.
Matplotlib: For visualizing the detection and recognition results.
Achievements:
Successfully built a system that could detect and recognize license plates from both images and video streams with high accuracy.
Achieved over 85% accuracy in detecting license plates and correctly recognizing characters in various lighting conditions and angles.
Developed a robust preprocessing pipeline that significantly improved the performance of OCR on noisy and low-contrast images.
Impact:
This ALPR system could be integrated into various real-world applications, such as automated toll collection, parking management systems, and security monitoring. The project demonstrated how OpenCV's image processing capabilities can be leveraged to solve practical computer vision problems.

Challenges Overcome:
Dealing with varying lighting conditions and plate orientations required careful tuning of preprocessing techniques like adaptive thresholding and contour filtering.
Tesseract OCR was sensitive to noise, so effective preprocessing and character segmentation were critical to improving recognition accuracy.