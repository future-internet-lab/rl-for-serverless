import uuid

'''    
Defines request state:
'''
class Request_States:
    In_Queue = 0
    In_System = 1
    Rejected = 2
    Done = 3 

class Request():
    """
    A class to represent a request in the RL-for-serverless environment.
    Attributes
    ----------
    type : int
        The type of the request.
    state : int, optional
        The state of the request (default is 0).
    max_queue_delay : int, optional
        The maximum queue delay for the request (default is 0).
    enq_ts : int, optional
        The enqueue timestamp of the request (default is 0).
    deq_ts : int
        The dequeue timestamp of the request (default is 0).
    exit_ts : int
        The exit timestamp of the request (default is 0).
    active_time : int, optional
        The active time of the request (default is 0).
    """
    def __init__(self, type: int, state: int = 0, 
                max_queue_delay: int = 0, enq_ts: int = 0, active_time: int = 0):
        self._uuid = uuid.uuid1()
        self.type = type
        self.max_queue_delay =  int(max_queue_delay) 
        self.enq_ts = int(enq_ts)
        self.deq_ts = 0
        self.exit_ts = 0
        self.state = state 
        self.active_time = int(active_time)
        
    def set_active_time(self, a):
        self.active_time = int(a)
        
    def set_time_out(self, a):
        self.max_queue_delay = int(a)
        
    def set_enq_ts(self, a):
        self.enq_ts = int(a) if a >= 0 else 0
        
    def set_deq_ts(self, a):
        self.deq_ts = int(a)
        
    def set_exit_ts(self, a):
        self.exit_ts = int(a)
    
    def set_state(self, state):
        self.state = state
