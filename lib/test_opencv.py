import cv2 as cv
import numpy as np

class CameraCapture:
    def __init__(self):
        self.cap = cv.VideoCapture(1)  # Используем камеру с индексом 1
        
    def start_capture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Не удалось захватить кадр с камеры")
                break
            
            processed_frame = self.preprocess(frame)
            contours, hierarchy = self.find_contours(processed_frame)
            
            # Рисуем только внутренние контуры на бинаризованном изображении для отображения
            contour_frame = cv.cvtColor(processed_frame, cv.COLOR_GRAY2BGR)
            
            for i, contour in enumerate(contours):
                if hierarchy[0, i, 3] != -1:  # Проверяем, является ли контур внутренним
                    # Выполняем аппроксимацию контура
                    perimeter = cv.arcLength(contour, True)
                    approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)
                    
                    # Если контур аппроксимируется к кругу (у круга 8 углов)
                    if len(approx) > 6:
                        cv.drawContours(contour_frame, [contour], -1, (255, 0, 0), 2)  # Рисуем внутренние контуры
            
            cv.imshow('Processed', contour_frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv.destroyAllWindows()
    
    def preprocess(self, frame):
        blurred = cv.GaussianBlur(frame, (5, 5), 0)
        gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return binary
    
    def find_contours(self, binary_frame):
        # Поиск контуров на бинаризованном изображении
        contours, hierarchy = cv.findContours(binary_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

# if __name__ == "__main__":
#     camera = CameraCapture()
#     camera.start_capture()
