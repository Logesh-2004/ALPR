import cv2
import pytesseract
import time
from datetime import datetime

# Path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dictionary to store vehicle entry and exit times
vehicle_times = {}

# Timer to add a delay for recognition
recognition_delay = 5  # in seconds
last_recognition_time = 0

# Detect license plate region
def detect_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)

    # Find contours in the image
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Assuming rectangular shape for license plates
            x, y, w, h = cv2.boundingRect(approx)
            plate = frame[y:y + h, x:x + w]  # Crop the plate
            return plate
    return None

# Preprocess the image for better OCR accuracy
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized_image = cv2.resize(thresholded, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoising(resized_image, h=30)
    return denoised_image

# Extract text from license plate
def extract_text_from_plate(plate_image):
    processed_image = preprocess_image(plate_image)
    
    # Adjust config for Indian number plates (only uppercase letters and digits)
    config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # OCR the processed image
    text = pytesseract.image_to_string(processed_image, config=config)
    
    # Clean up detected text
    plate_text = ''.join(filter(str.isalnum, text))
    
    # Ensure the text matches a valid plate format (e.g., TN07AB1234)
    if len(plate_text) >= 8 and len(plate_text) <= 10:
        return plate_text.strip()
    return None

# Log entry or exit time
def log_entry_exit(plate_text):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if plate_text not in vehicle_times:
        # Log entry time
        vehicle_times[plate_text] = {"entry": current_time, "exit": None}
        print(f"Vehicle {plate_text} entered at {current_time}")
    else:
        # Log exit time
        if vehicle_times[plate_text]["exit"] is None:
            vehicle_times[plate_text]["exit"] = current_time
            print(f"Vehicle {plate_text} exited at {current_time}")

# Main function to capture and process video feed
def main():
    global last_recognition_time
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Draw a green rectangle in the middle of the screen for plate alignment
        height, width, _ = frame.shape
        # Define the rectangle position (centered)
        x1, y1 = int(width * 0.3), int(height * 0.4)
        x2, y2 = int(width * 0.7), int(height * 0.6)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show camera feed with the green rectangle
        cv2.imshow("ALPR System", frame)

        # Capture the frame when 'c' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Press 'c' to capture the frame
            current_time = time.time()

            # Check delay for recognition
            if current_time - last_recognition_time >= recognition_delay:
                # Only capture the part of the frame inside the rectangle
                plate_image = frame[y1:y2, x1:x2]

                # Perform detection and OCR
                plate_text = extract_text_from_plate(plate_image)
                
                if plate_text:
                    log_entry_exit(plate_text)
                    last_recognition_time = current_time
                else:
                    print("No valid plate detected.")
        
        # Press 'q' to quit
        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the recorded entry and exit times
    print("\nVehicle Log:")
    for plate, times in vehicle_times.items():
        print(f"Vehicle {plate}: Entry = {times['entry']}, Exit = {times['exit']}")

if __name__ == "__main__":
    main()
