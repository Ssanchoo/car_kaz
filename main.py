import cv2
import os
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from webcam import Webcam  # Импортируем класс Webcam


def carplate_extract(image, carplate_haar_cascade):
    """Извлекает номерной знак с помощью каскада Haar."""
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    carplate_img = None
    for x, y, w, h in carplate_rects:
        carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]

    return carplate_img


def enlarge_img(image, scale_percent):
    """Увеличивает изображение на заданный процент."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image


def process_frame(frame, carplate_haar_cascade):
    """Обрабатывает один кадр видеопотока."""
    # Извлекаем номерной знак с кадра
    carplate_img = carplate_extract(frame, carplate_haar_cascade)

    if carplate_img is not None and carplate_img.size > 0:
        # Увеличиваем изображение номерного знака
        carplate_img = enlarge_img(carplate_img, 150)

        # Преобразуем в серые тона для улучшения распознавания
        carplate_img_gray = cv2.cvtColor(carplate_img, cv2.COLOR_RGB2GRAY)

        # Распознаем номер с помощью pytesseract
        car_plate_number = pytesseract.image_to_string(
            carplate_img_gray,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        return frame, car_plate_number
    else:
        return frame, None


def main():
    # Путь к файлам
    haar_cascade_path = r'./haar_cascades/haarcascade_russian_plate_number.xml'

    # Проверка существования файла Haar Cascade
    if not os.path.exists(haar_cascade_path):
        raise FileNotFoundError(f"Не удалось найти файл Haar Cascade по пути: {haar_cascade_path}")

    # Загружаем каскад
    carplate_haar_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if carplate_haar_cascade.empty():
        raise FileNotFoundError("Не удалось загрузить Haar Cascade.")

    # Инициализация видеопотока с камеры
    webcam = Webcam()

    print("Запуск видеопотока...")

    while True:
        # Захват кадра с камеры
        ret, frame = webcam.get_current_frame()

        if not ret:
            print("Ошибка: не удается захватить кадр.")
            break

        # Обрабатываем кадр
        frame_with_plate, car_plate_number = process_frame(frame, carplate_haar_cascade)

        if car_plate_number:
            print(f"Номер авто: {car_plate_number}")

        # Отображаем кадр с возможным номерным знаком
        cv2.imshow("Frame", frame_with_plate)

        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
