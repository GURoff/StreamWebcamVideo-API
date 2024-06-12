import cv2
import pytesseract
import threading
import time

class RealTimeTextRecognition:
    def __init__(self, tesseract_cmd_path):
        # Provide the path to the Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        # Initializing video capture from a webcam
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            raise Exception("Failed to open webcam")

        self.running = True
        self.prev_text = None

    def __del__(self):
        # Freeing up webcam resource
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def capture_and_process_image(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Image capture error")
                break

            # Converting an Image to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Using Tesseract for OCR
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
            boxes = pytesseract.image_to_boxes(gray, config='--oem 3 --psm 6')

            # Drawing rectangles around text
            h, w = gray.shape
            for b in boxes.splitlines():
                b = b.split(' ')
                x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                # Converting coordinates for display on an image
                cv2.rectangle(frame, (x, h - y), (x2, h - y2), (0, 255, 0), 2)

            # Checking if the current text matches the previous one
            if text.strip() == self.prev_text:
                continue
            # Saving the current text for the next iteration
            self.prev_text = text.strip()
            
            # Output of recognized text
            print(f"Recognized text: {text.strip()}")

            # Image display
            cv2.imshow('Camera', frame)

            # Quitting when pressing 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
            
            # Adding delay
            time.sleep(0.5)

    def start(self):
        # Starting a thread to process images
        threading.Thread(target=self.capture_and_process_image).start()
