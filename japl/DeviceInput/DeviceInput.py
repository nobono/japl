import threading

from queue import Queue

from inputs import get_gamepad
from inputs import get_mouse

# ---------------------------------------------------



class DeviceInput:

    def __init__(self, device_type: str = "mouse") -> None:

        self.devices = {
                "mouse": self.get_mouse_input,
                "gamepad": self.get_gamepad_input,
                }

        assert device_type in self.devices
        self.queue = Queue()
        self.device_data = [0, 0, 0, 0] # [x-left, y-left, x-right, y-right]
        self.device_thread = threading.Thread(target=self.devices[device_type], args=(self.queue,))
        self.running = False


    def test(self, N: int = 10) -> None:
        self.start()
        for _ in range(N):
            data = self.queue.get()
            print(data)
        self.stop()


    def start(self) -> None:
        self.running = True
        self.device_thread.start()


    def stop(self) -> None:
        self.running = False
        self.device_thread.join()


    def __del__(self) -> None:
        if self.running:
            self.stop()


    def empty(self) -> bool:
        return self.queue.empty()


    def get(self) -> list:
        while not self.queue.empty():
            self.device_data = self.queue.get()
        return self.device_data


    def get_mouse_input(self, queue: Queue) -> None:
        _norm = 5.0
        x, y = (0, 0)
        while self.running:
            events = get_mouse()

            for event in events:
                # print(event.ev_type, event.code, event.state)
                if event.code == "REL_X":
                    x = event.state / _norm
                elif event.code == "REL_Y":
                    y = event.state / _norm

            mouse_data = [float(x), -float(y), 0, 0]

            queue.put(mouse_data)


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
