import zmq

class MsgPublisher:
    def __init__(self, port) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f'tcp://*:{port}')

    def send(self, topic:str, msg:dict):
        self.socket.send_string(topic, flags=zmq.SNDMORE)
        self.socket.send_json(msg)

class MsgSubscriber:
    def __init__(self, ip, port, topic) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f'tcp://{ip}:{port}')
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)


    def recv(self):
        topic = self.socket.recv_string()
        msg = self.socket.recv_json()
        return msg