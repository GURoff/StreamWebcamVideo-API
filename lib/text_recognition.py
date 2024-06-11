import cv2
import pytesseract

class RealTimeTextRecognition:
    def __init__(self, tesseract_cmd_path):
        # Provide the path to the Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        # Initializing video capture from a webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Failed to open webcam")

    def __del__(self):
        # Freeing up webcam resource
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def capture_and_process_image(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Image capture error")
                break

            # Flip the image horizontally
            flipped_frame = cv2.flip(frame, 1)

            # Image preprocessing
            gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Display a preprocessed image for debugging
            cv2.imshow('Preprocessed Image', thresh)

            # Using Tesseract for OCR
            text = pytesseract.image_to_string(thresh, config='--psm 6')

            # Output of recognized text
            print(f"Recognized text: {text.strip()}")

            # Image display
            cv2.imshow('Camera', frame)

            # Quitting when pressing 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    # Provide the path to the Tesseract executable
    tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    recognizer = RealTimeTextRecognition(tesseract_cmd_path)
    recognizer.capture_and_process_image()
