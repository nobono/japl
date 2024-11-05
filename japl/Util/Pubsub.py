import queue
from warnings import warn



class Subscriber:
    def __init__(self, name):
        self.name = name
        self.msg_buffer = []

    def receive(self, message: dict):
        self.msg_buffer.append(message)


class Publisher:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.subscribers = {}

    def subscribe(self, subscriber: Subscriber):
        # restrict one channel per subscriber
        if subscriber.name in self.subscribers:
            raise Exception(f"channel {subscriber.name} already assigned to\
                            publisher")
        else:
            self.subscribers.update({subscriber.name: subscriber})

    def publish(self, message: dict):
        # self.message_queue.put(message)
        for subscriber in self.subscribers.values():
            subscriber.receive(message)

    def publish_to(self, channels: str|list[str], message: dict):
        if hasattr(channels, "__len__"):
            for channel in channels:
                self.subscribers[channel].receive(message)
        else:
            self.subscribers[channels].receive(message)
        # self.message_queue.put(message)
            # subscriber.receive(message)

# publisher = Publisher()

# subscriber_1 = Subscriber("Subscriber 1")
# subscriber_2 = Subscriber("Subscriber 2")

# publisher.subscribe(subscriber_1)
# publisher.subscribe(subscriber_2)

# publisher.publish("Hello World")
