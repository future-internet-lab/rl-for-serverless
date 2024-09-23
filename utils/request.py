import numpy as np
import uuid

'''
Define resource usage for request type:
 request_type: [RAM, CPU, Power]
    
'''    
Request_Resource_Usage = np.array([np.array([10, 10, 100]),
                                   np.array([20, 20, 200]),
                                   np.array([30, 30, 300]),
                                   np.array([40, 40, 400])])

Request_active_time = np.array([240, 360, 480, 600])

def ran_norm_gen(mean, std_dev):
    # Generate a random value following normal distribution
    value = np.random.normal(loc=mean, scale=std_dev)
    # Round to the nearest integer
    int_value = round(value)
    # Ensure the value is greater than 0
    positive_int_value = max(1, int_value)  # Ensures the value is at least 1
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
        self.resource_usage = Request_Resource_Usage[self.type]
        
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

def generate_requests(queue, current_time, size, avg_requests_per_second, timeout, max_rq_active_time):
    rng = np.random.default_rng()
    num_requests = rng.poisson(avg_requests_per_second)
    num_new_rq = np.zeros(size,dtype=np.int32)
    
    for i in range(num_requests):
        type = rng.integers(0, size)
        if max_rq_active_time["type"] == "random":
            active_time = ran_norm_gen(max_rq_active_time["value"][type], max_rq_active_time["value"][type]/10)
        else:
            active_time = max_rq_active_time["value"][type] if max_rq_active_time["value"][type] else Request_active_time[type]
        request = Request(type=type, in_queue_time=int(current_time), timeout=timeout[type], active_time=active_time)    
        queue[type].append(request)
        num_new_rq[type] += 1
    return num_new_rq
