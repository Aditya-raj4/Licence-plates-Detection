import cv2
import pytesseract
import matplotlib.pyplot as plt

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image):
    """ Preprocess the image: grayscale, blur, and edge detection. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)  # Adjusted for better edge detection
    return edged


def find_license_plate_contour(edged):
    """ Find contours in the edged image and filter for license plate region. """
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate_contour = None
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 2 < aspect_ratio < 5 and area > max_area:  # Filter by aspect ratio and size
            plate_contour = contour
            max_area = area
    return plate_contour


def extract_license_plate(image, contour):
    """ Extract and preprocess the license plate region from the image. """
    x, y, w, h = cv2.boundingRect(contour)
    license_plate = image[y:y + h, x:x + w]
    return license_plate


def perform_ocr(license_plate):
    """ Perform OCR on the license plate image. """
    # Convert to grayscale and threshold
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Tesseract to extract the text
    text = pytesseract.image_to_string(binary_plate, config='--psm 8')
    return text.strip()


def main():
    # Load the uploaded image
    image_path = 'image/License-plate-india-header-1920x730.jpg'
    image = cv2.imread(image_path)

    # Preprocess the image
    edged_image = preprocess_image(image)

    # Find the license plate contour
    plate_contour = find_license_plate_contour(edged_image)

    if plate_contour is not None:
        # Extract the license plate from the image
        license_plate = extract_license_plate(image, plate_contour)

        # Perform OCR to read the license plate text
        plate_text = perform_ocr(license_plate)

        # Draw the contour and show the detected license plate
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'License Plate: {plate_text}')
        plt.show()
    else:
        print("License plate not detected.")


if __name__ == "__main__":
    main()