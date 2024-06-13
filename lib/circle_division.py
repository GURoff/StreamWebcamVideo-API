import cv2
import numpy as np
import threading
import time
import math

class RealTimeTextRecognition:
    def __init__(self):
        # Инициализация видеозахвата с веб-камеры
        self.cap = cv2.VideoCapture(1)  # Используем камеру с индексом 1 (обычно это встроенная веб-камера)
        if not self.cap.isOpened():
            raise Exception("Failed to open webcam")

        self.running = True
        self.fixed_value = None  # Переменная для фиксации значения

    def __del__(self):
        # Освобождение ресурсов
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def detect_arrow(self, image):
        # Преобразование изображения в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применение размытия для снижения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Применение операции Canny для выделения границ
        edges = cv2.Canny(blurred, 50, 150)

        # Поиск кругов на изображении с помощью метода HoughCircles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=0, maxRadius=0)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                # Нарисовать круг на изображении
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                # Ограничить область интереса (ROI) кругом
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, 255, -1)
                masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

                # Поиск контуров внутри круга
                contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Находим самый длинный контур
                    longest_contour = max(contours, key=lambda contour: cv2.arcLength(contour, True))

                    # Находим ограничивающий прямоугольник для контура
                    bx, by, bw, bh = cv2.boundingRect(longest_contour)

                    # Находим центр ограничивающего прямоугольника
                    center = (bx + bw // 2, by + bh // 2)

                    # Находим ориентацию контура
                    angle = cv2.fitEllipse(longest_contour)[-1]

                    return center, angle, (x, y, r)

        return None, None, None

    def recognize_value(self, angle):
        # Шкала значений на барометре (в градусах)
        scale_values = {
            (-90, -45): 'Stormy',
            (-45, 45): 'Normal',
            (45, 90): 'Sunny'
        }

        # Определяем значение на основе угла
        for (min_angle, max_angle), value in scale_values.items():
            if min_angle <= angle <= max_angle:
                return value

        return 'Unknown'

    def detect_markings(self, image, circle):
        x, y, r = circle
        num_divisions = 41  # Количество делений на барометре

        # Рассчитываем угол между делениями
        angle_step = 360 / num_divisions

        # Отображаем нулевой угол (вертикальная линия)
        zero_angle = 0  # Угол вертикальной линии (12 часов)
        zero_radian = math.radians(zero_angle)
        zero_pt1 = (x + int(r * math.cos(zero_radian)), y + int(r * math.sin(zero_radian)))
        cv2.line(image, (x, y), zero_pt1, (0, 255, 255), 2)

        # Отображаем первую линию как 100
        first_line_angle = 40  # Угол первой линии
        first_line_radian = math.radians(first_line_angle)
        first_line_pt1 = (x + int(r * math.cos(first_line_radian)), y + int(r * math.sin(first_line_radian)))
        cv2.line(image, (x, y), first_line_pt1, (0, 0, 255), 2)
        cv2.putText(image, '100', first_line_pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Отображаем последнюю линию как 0
        last_line_angle = 135  # Угол последней линии
        last_line_radian = math.radians(last_line_angle)
        last_line_pt1 = (x + int(r * math.cos(last_line_radian)), y + int(r * math.sin(last_line_radian)))
        cv2.line(image, (x, y), last_line_pt1, (0, 0, 255), 2)
        cv2.putText(image, '0', last_line_pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def capture_and_process_image(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Image capture error")
                break

            # Обнаружение стрелки на барометре
            center, angle, circle = self.detect_arrow(frame)

            # Если стрелка обнаружена
            if center is not None and angle is not None:
                # Распознавание значения на барометре
                value = self.recognize_value(angle)

                # Если значение "Normal" найдено и еще не зафиксировано
                if value == "Normal" and self.fixed_value is None:
                    self.fixed_value = value
                    print(f'Barometer fixed value: {value}')

                # Отображение текущего значения, если оно не зафиксировано
                if self.fixed_value is None:
                    print(f'Barometer value: {value}')
                    cv2.putText(frame, f'Value: {value}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # Отображение зафиксированного значения
                    cv2.putText(frame, f'Fixed Value: {self.fixed_value}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Обнаружение делений на барометре
                self.detect_markings(frame, circle)

            # Отображение изображения с камеры
            cv2.imshow('Camera', frame)

            # Выход при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

            # Добавление небольшой задержки
            time.sleep(0.5)  # Измените это значение по вашему усмотрению

    def start(self):
        # Запуск потока для обработки изображений
        threading.Thread(target=self.capture_and_process_image).start()

#  if __name__ == "__main__":
#      recognizer = RealTimeTextRecognition()
#     recognizer.start()
