import threading

from queue import Queue

from inputs import get_gamepad
from inputs import get_mouse

# ---------------------------------------------------



class DeviceInput:
    def __init__(self) -> None:
        self.queue = Queue()
        self.device_data = [0, 0, 0, 0] # [x-left, y-left, x-right, y-right]
        self.mouse_thread = threading.Thread(target=self.get_mouse_input, args=(self.queue,))
        self.thread_running = False


    def start(self) -> None:
        self.thread_running = True
        self.mouse_thread.start()


    def __del__(self) -> None:
        if self.thread_running:
            self.mouse_thread.join()
            self.thread_running = False


    def empty(self) -> bool:
        return self.queue.empty()


    def get(self) -> list:
        if not self.queue.empty():
            self.device_data = self.queue.get()
        return self.device_data


    def get_mouse_input(self, queue: Queue) -> None:

        _norm = (2**16 / 2)
        lx, rx = (0, 0)
        ly, ry = (0, 0)
        while True:
            events = get_mouse()

            for event in events:
                # print(event.ev_type, event.code, event.state)
                if event.code == "ABS_X":
                    lx = event.state / _norm
                elif event.code == "ABS_RX":
                    rx = event.state / _norm
                elif event.code == "ABS_Y":
                    ly = event.state / _norm
                elif event.code == "ABS_RY":
                    ry = event.state / _norm

            mouse_data = [lx, -ly, rx, -ry]

            queue.put_nowait(mouse_data)


    def get_gamepad_input(self, queue: Queue) -> None:
        _norm = (2**16 / 2)
        lx, rx = (0, 0)
        ly, ry = (0, 0)
        while True:
            events = get_gamepad()

            for event in events:
                # print(event.ev_type, event.code, event.state)
                if event.code == "ABS_X":
                    lx = event.state / _norm
                elif event.code == "ABS_RX":
                    rx = event.state / _norm
                elif event.code == "ABS_Y":
                    ly = event.state / _norm
                elif event.code == "ABS_RY":
                    ry = event.state / _norm

            mouse_data = [lx, -ly, rx, -ry]

            queue.put_nowait(mouse_data)
