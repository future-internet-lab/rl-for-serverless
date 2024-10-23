import numpy as np
import uuid
import rlss_env.profiling as profiling

'''    
Defines request state:
'''
class Request_States:
    In_Queue = 0
    In_System = 1
    Time_Out = 2
    Done = 3 

def ran_norm_gen(mean, std_dev):
    value = np.random.normal(loc=mean, scale=std_dev)
    int_value = round(value)
    positive_int_value = max(1, int_value)
    return positive_int_value

class Request():
    def __init__(self, type: int, state: int = 0, 
                timeout: int = 0, in_queue_time: int = 0, active_time: int = 0):
        self._uuid = uuid.uuid1()
        self.type = type
        self.time_out =  timeout 
        self.in_queue_time = in_queue_time
        self.in_system_time = 0
        self.out_system_time = 0
        self.state = state 
        self.resource_usage = None
        self.active_time = active_time
        self.set_resource_usage()

    def set_resource_usage(self):
        self.resource_usage = profiling.REQ_RES_USAGE[self.type]
        
    def set_active_time(self, a):
        self.active_time = a
        
    def set_time_out(self, a):
        self.time_out = a
        
    def set_in_queue_time(self, a):
        self.in_queue_time = a if a >= 0 else 0
        
    def set_in_system_time(self, a):
        self.in_system_time = a
        
    def set_out_system_time(self, a):
        self.out_system_time = a
    
    def set_state(self, state):
        self.state = state

# Generate requests for the queue
def generate_requests(queue, current_time, size, avg_requests_per_second, timeout, max_rq_active_time):
    rng = np.random.default_rng()
    num_requests = rng.poisson(avg_requests_per_second)  # Generate number of requests
    num_new_requests = np.zeros(size, dtype=np.int32)    # Initialize new request count

    # Process each request
    for _ in range(num_requests):
        request_type = rng.integers(0, size)  # Random request type
        active_time = determine_active_time(request_type, max_rq_active_time)  # Get active time
        request = create_request(request_type, current_time, timeout, active_time)  # Create request
        queue[request_type].append(request)  # Add request to queue
        num_new_requests[request_type] += 1   # Increment count for this type
    
    return num_new_requests

# Helper to determine active time based on request type
def determine_active_time(request_type, max_rq_active_time):
    if max_rq_active_time["type"] == "random":
        return ran_norm_gen(max_rq_active_time["value"][request_type], max_rq_active_time["value"][request_type]/2)
    else:
        return max_rq_active_time["value"][request_type] or profiling.REQ_ACTIVE_TIME[request_type]

# Helper to create a new request
def create_request(request_type, current_time, timeout, active_time):
    return Request(type=request_type, in_queue_time=int(current_time), timeout=timeout[request_type], active_time=active_time)
