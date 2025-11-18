from queue import CircularQueue, QueueFullError
from request import Request_States
from container import Container_States

class Controller():
    def __init__(self, env, containers, queue_len=100000, logger=None):
        self.type = type
        self.env = env
        self.queue = CircularQueue(queue_len)
        self.containers = containers
        self.current_warm_containers = []
        
def receive_request(self, request):
        """Nhận request vào hàng đợi"""
        try:
            request.state = Request_States.In_Queue
            self.queue.enqueue(request)
            print(f"[{self.env.now:.2f}] Request {request.id} vào hàng đợi")
            yield self.env.process(self.request_timeout(request))
            next_request = self.queue.dequeue()
            self.current_warm_containers = self.get_warm_containers()
            yield env.process(self.process_request(next_request))
                
        except QueueFullError:
            print(f"[{self.env.now:.2f}] Request {request.id} bị từ chối - hàng đợi đầy")
            return

def process_request(self, request):
    if request.state != Request_States.Rejected:
        container = self.current_warm_containers.pop(0)
        yield self.env.process(container.receive_request(request))
    if len(self.current_warm_containers) != 0 or not self.queue.is_empty():
        next_request = self.queue.dequeue()
        yield self.env.process(self.process_request(next_request))

def get_warm_containers(self):
        """Trả về danh sách các container ở trạng thái 'Warm'"""
        return [container for container in self.containers if container.get_state() == Container_States.Warm_CPU]
    
def request_timeout(self, request):
        """Kiểm tra timeout cho request"""
        yield self.env.timeout(request.timeout)
        request.state = Request_States.Rejected
            
            