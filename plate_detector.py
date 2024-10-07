import cv2
import pytesseract

# Set path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_license_plate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find the license plate
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Adjust these values based on your needs
        if 2 < aspect_ratio < 5:  # Adjust aspect ratio for plates
            plate_image = gray[y:y + h, x:x + w]
            return plate_image  # Return the cropped plate image

    return None  # If no plate was detected

def extract_text_from_plate(plate_image):
    if plate_image is not None:
        # Resize the plate image for better detection
        plate_image = cv2.resize(plate_image, None, fx=2, fy=2)

        # Apply denoising to clean the image
        denoised = cv2.fastNlMeansDenoising(plate_image, None, 30, 7, 21)

        # Apply thresholding to binarize the image
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Use morphological operations to enhance the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Use pytesseract to extract text
        text = pytesseract.image_to_string(morph, config='--psm 7')
        return text.strip()
    
    return "No plate detected"
