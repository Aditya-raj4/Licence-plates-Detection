import cv2
import pytesseract

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image):
    """ Preprocess the image: grayscale, blur, and edge detection. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Adjusted kernel size
    edged = cv2.Canny(blurred, 30, 150)  # Adjusted thresholds
    return edged


def find_license_plate_contour(edged):
    """ Find contours in the edged image and filter for license plate region. """
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate_contour = None
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)  # Debugging
        if 2 < aspect_ratio < 5 and area > 1000:  # Adjusted filters
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
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary_plate, config='--psm 8')  # Try --psm 7 or --psm 11
    return text.strip()


def main():
    # Initialize video capture (0 for webcam, or provide a video file path)
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or replace with a video file path

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Preprocess the frame
        edged_image = preprocess_image(frame)

        # Find the license plate contour
        plate_contour = find_license_plate_contour(edged_image)

        if plate_contour is not None:
            # Extract the license plate from the frame
            license_plate = extract_license_plate(frame, plate_contour)

            # Perform OCR to read the license plate text
            plate_text = perform_ocr(license_plate)

            # Draw the contour and display the detected license plate text
            cv2.drawContours(frame, [plate_contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, f'License Plate: {plate_text}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the extracted license plate
            cv2.imshow('License Plate', license_plate)

        # Display the preprocessed edged image
        cv2.imshow('Edged Image', edged_image)

        # Display the frame
        cv2.imshow('Live License Plate Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()