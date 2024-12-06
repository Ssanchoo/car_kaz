import cv2

class Webcam:
    def __init__(self):
        # Инициализация видеопотока с камеры
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise Exception("Не удалось открыть камеру.")

    def get_current_frame(self):
        """Получает текущий кадр с видеопотока."""
        ret, frame = self.video_capture.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            return ret, None
        return ret, frame

    def release(self):
        """Освобождает видеопоток."""
        self.video_capture.release()

    def show_frame(self, frame):
        """Отображает текущий кадр."""
        cv2.imshow('Live Frame', frame)
