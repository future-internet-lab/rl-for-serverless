from rlss_env.queue import CircularQueue, QueueFullError
'''    
Defines symbols in a state machine:
    - N  = Null
    - L0 = Cold
    - L1 = Warm Disk
    - L2 = Warm CPU
    - A  = Active
'''
class Container_States:
    Null = 0
    Cold = 1
    Warm_Disk = 2
    Warm_CPU = 3
    Active = 4
    State_Name = ["Null", "Cold", "Warm Disk", "Warm CPU", "Active"]

    
class Container():
    def __init__(self, env, container_id, type, res_profile, idle_time_window, state=Container_States.Null, queue_len=1, exporter=None):
        self.type = type
        self.env = env
        self.container_id = container_id
        self.state = state
        self.res_profile = res_profile
        self.idle_time_window = idle_time_window
        self.current_res_usage = res_profile[state]
        self.queue = CircularQueue(queue_len)
        self.exporter = exporter
            
    def receive_request(self, request):
        """Nhận request vào hàng đợi"""
        try:
            self.queue.enqueue(request)
            next_request = self.queue.dequeue()
            yield env.process(self.process_request(next_request))
                
        except QueueFullError:
            return 
        
    def process_request(self, request):
        """Xử lý request"""
        self.state = Container_States.Active
        self.current_res_usage = self.res_profile[self.state]
        yield env.timeout(request.active_duration)
        
        if not self.queue.is_empty():
            next_request = self.queue.dequeue()
            yield env.process(self.process_request(next_request))
        
        self.state = Container_States.Warm_CPU    
        yield env.timeout(self.idle_time_window)
        self.state = Container_States.Warm_Disk
        
    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state
        self.current_res_usage = self.res_profile[state]