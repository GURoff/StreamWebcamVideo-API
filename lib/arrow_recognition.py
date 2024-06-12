import cv2
import numpy as np

def detect_arrow(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение размытия для снижения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применение операции Canny для выделения границ
    edges = cv2.Canny(blurred, 50, 150)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры найдены
    if contours:
        # Находим самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)

        # Находим ограничивающий прямоугольник для контура
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Находим центр ограничивающего прямоугольника
        center = (x + w // 2, y + h // 2)

        # Находим ориентацию контура
        angle = cv2.fitEllipse(largest_contour)[-1]

        return center, angle

    return None, None

def recognize_value(angle):
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

if __name__ == "__main__":
    # Чтение изображения с барометром
    image = cv2.imread('barometer_image.jpg')

    # Детектирование стрелки и определение угла
    center, angle = detect_arrow(image)

    # Если стрелка обнаружена
    if center is not None and angle is not None:
        # Распознавание значения
        value = recognize_value(angle)

        # Отображение результата
        cv2.putText(image, f'Value: {value}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Barometer', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Arrow not detected")
