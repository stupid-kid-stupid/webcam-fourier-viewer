"""Display a live webcam preview and its Fourier spectra."""

import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


FrameResult = tuple[np.ndarray, np.ndarray, np.ndarray]


def calculate_frame_result(frame: np.ndarray) -> FrameResult:
    """Return the RGB frame and RGB pseudocolor power and phase spectra."""
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    spectrum = np.fft.fftshift(np.fft.fft2(grayscale))

    power = np.log1p(np.square(np.abs(spectrum)))
    power_gray = cv2.normalize(
        power, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    power_bgr = cv2.applyColorMap(power_gray, cv2.COLORMAP_TURBO)
    power_image = cv2.cvtColor(power_bgr, cv2.COLOR_BGR2RGB)

    phase = np.angle(spectrum)
    phase_gray = np.clip((phase + np.pi) * (255.0 / (2.0 * np.pi)), 0, 255)
    phase_gray = phase_gray.astype(np.uint8)
    phase_bgr = cv2.applyColorMap(phase_gray, cv2.COLORMAP_TURBO)
    phase_image = cv2.cvtColor(phase_bgr, cv2.COLOR_BGR2RGB)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame, power_image, phase_image


class CameraApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("摄像头实时图像与傅里叶频谱")
        self.root.geometry("1350x720")
        self.root.minsize(900, 480)

        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.poll_after_id: str | None = None
        self.result_lock = threading.Lock()
        self.latest_result: FrameResult | None = None
        self.worker_messages: queue.Queue[str] = queue.Queue()
        self.photo_images: list[ImageTk.PhotoImage | None] = [None, None, None]
        self.status = tk.StringVar(value="点击“开启摄像头”开始预览")
        self.image_labels: list[ttk.Label] = []

        panels = ttk.Frame(root, padding=(12, 12, 12, 8))
        panels.pack(fill=tk.BOTH, expand=True)
        panels.rowconfigure(0, weight=1)
        panel_titles = ("摄像头实时画面", "傅里叶功率谱", "傅里叶相位谱")
        column_weights = (2, 1, 1)

        for column, (title, weight) in enumerate(zip(panel_titles, column_weights)):
            panels.columnconfigure(column, weight=weight, uniform="panel")
            panel = ttk.LabelFrame(panels, text=title, padding=8)
            panel.grid(row=0, column=column, sticky="nsew", padx=5)
            panel.rowconfigure(0, weight=1)
            panel.columnconfigure(0, weight=1)
            label = ttk.Label(panel, text="摄像头尚未开启", anchor=tk.CENTER)
            label.grid(row=0, column=0, sticky="nsew")
            self.image_labels.append(label)

        controls = ttk.Frame(root, padding=(12, 0, 12, 12))
        controls.pack(fill=tk.X)
        ttk.Label(controls, textvariable=self.status).pack(side=tk.LEFT)
        self.start_button = ttk.Button(
            controls,
            text="开启摄像头",
            command=self.start_camera,
        )
        self.start_button.pack(side=tk.RIGHT, padx=(6, 0))
        self.stop_button = ttk.Button(
            controls,
            text="关闭摄像头",
            command=self.stop_camera,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.RIGHT)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def start_camera(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        self._clear_worker_messages()
        with self.result_lock:
            self.latest_result = None

        self.stop_event = threading.Event()
        self.worker = threading.Thread(
            target=self._capture_frames,
            args=(self.stop_event,),
            daemon=True,
        )
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status.set("正在连接摄像头...")
        self._set_placeholder("正在连接摄像头...")
        self.worker.start()
        self._schedule_poll()

    def stop_camera(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self.status.set("正在关闭摄像头...")
        self._set_placeholder("摄像头已关闭")
        with self.result_lock:
            self.latest_result = None
        self._schedule_poll()

    def _capture_frames(self, stop_event: threading.Event) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not capture.isOpened():
                self.worker_messages.put(
                    "无法打开摄像头，请检查连接和系统摄像头权限。"
                )
                return

            if stop_event.is_set():
                return
            self.worker_messages.put("摄像头已开启")

            while not stop_event.is_set():
                success, frame = capture.read()
                if stop_event.is_set():
                    break
                if not success:
                    self.worker_messages.put(
                        "读取摄像头画面失败，请重新开启摄像头。"
                    )
                    break

                result = calculate_frame_result(frame)
                if stop_event.is_set():
                    break
                with self.result_lock:
                    self.latest_result = result
        except Exception as error:
            self.worker_messages.put(f"摄像头处理失败：{error}")
        finally:
            if capture is not None:
                capture.release()

    def _schedule_poll(self) -> None:
        if self.poll_after_id is None:
            self.poll_after_id = self.root.after(15, self._poll_worker)

    def _poll_worker(self) -> None:
        self.poll_after_id = None
        while True:
            try:
                self.status.set(self.worker_messages.get_nowait())
            except queue.Empty:
                break

        with self.result_lock:
            result = self.latest_result
            self.latest_result = None
        if result is not None:
            for index, image in enumerate(result):
                self._show_image(index, image)

        if self.worker is not None and self.worker.is_alive():
            self._schedule_poll()
            return

        if self.stop_event is not None and self.stop_event.is_set():
            self.status.set("摄像头已关闭")
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def _show_image(self, index: int, pixels: np.ndarray) -> None:
        label = self.image_labels[index]
        width = max(label.winfo_width() - 8, 1)
        height = max(label.winfo_height() - 8, 1)
        image = Image.fromarray(pixels)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        self.photo_images[index] = photo

    def _set_placeholder(self, message: str) -> None:
        for label in self.image_labels:
            label.configure(image="", text=message)
        self.photo_images = [None, None, None]

    def _clear_worker_messages(self) -> None:
        while True:
            try:
                self.worker_messages.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        if self.poll_after_id is not None:
            self.root.after_cancel(self.poll_after_id)
            self.poll_after_id = None
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    CameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
