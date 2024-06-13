import cv2
import numpy as np
import threading

class RealTimeTextRecognition:
    def __init__(self):
        # Инициализация видеозахвата с веб-камеры
        self.cap = cv2.VideoCapture(1)  # Используем камеру с индексом 1 (первая доступная камера)
        if not self.cap.isOpened():
            raise Exception("Failed to open webcam")

        self.running = True
        self.capture_triggered = False  # Флаг для фиксации изображения

    def __del__(self):
        # Освобождение ресурсов
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def detect_barometer(self, image):
        # Преобразование изображения в оттенки серого и применение размытия для снижения шума
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 2)

        # Применение операции Canny для выделения границ
        edges = cv2.Canny(blurred, 50, 150)

        # Поиск кругов на изображении с помощью метода HoughCircles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=60, minRadius=50, maxRadius=300)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Нарисовать круг на изображении
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                return (x, y, r)
        return None

    def detect_edges(self, image, circle):
        x, y, r = circle

        # Создаем маску для зеленой окружности ROI
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

        # Применяем маску к изображению и выделяем границы
        masked_image = cv2.bitwise_and(image, mask)
        gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        blurred_masked = cv2.GaussianBlur(gray_masked, (15, 15), 2)  # Увеличен размер ядра для размытия
        edges = cv2.Canny(blurred_masked, 50, 150)

        # Применение медианного фильтра для дальнейшего уменьшения шумов
        blurred_masked = cv2.medianBlur(gray_masked, 5)
        edges = cv2.Canny(blurred_masked, 50, 150)

        kernel = np.ones((3, 3), np.uint8)  # Морфологическое ядро
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # Морфологическая операция закрытия
       # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)   # Морфологическая операция открытия

        image_with_edges = np.copy(image)
        image_with_edges[edges != 0] = [0, 0, 255]  # Красный цвет для граней

        # Проверяем наличие контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and not self.capture_triggered:
            # Если обнаружены контуры, фиксируем изображение и отображаем текст
            cv2.putText(image, 'Edges Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.capture_triggered = True

        # Отображение граней на исходном изображении
        image_with_edges = np.copy(image)
        image_with_edges[edges != 0] = [0, 0, 255]  # Красный цвет для граней
        cv2.imshow('Edges', image_with_edges)

    def capture_and_process_image(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Image capture error")
                break

            # Обнаружение барометра на изображении
            circle = self.detect_barometer(frame)

            # Если круг (барометр) обнаружен
            if circle is not None:
                # Детекция граней внутри зеленой окружности
                self.detect_edges(frame, circle)

            # Выход при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def start(self):
        # Запуск потока для обработки изображений
        threading.Thread(target=self.capture_and_process_image).start()

# if __name__ == "__main__":
#     recognizer = RealTimeTextRecognition()
#     recognizer.start()
