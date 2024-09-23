import numpy as np

import gymnasium as gym
from gymnasium import spaces

import utils.request as rq
from utils.request import Request_Resource_Usage
import itertools
import math
import time

def compute_formula(num_box, num_ball):
    numerator = math.factorial(num_box + num_ball - 1)
    denominator = math.factorial(num_box - 1) * math.factorial(num_ball)
    return int(numerator/denominator)

def uniform_random_time(a, b, step=1):
    possible_values = np.arange(a, b + step, step)
    return np.random.choice(possible_values)


'''
Define an index corresponding to the action that changes the container's state:
    - Destination state is changed from the original state: 1
    - Source state is changed to another state: -1
    - State is not changed: 0
'''
class Resource_Type:
    RAM = 0
    CPU = 1
    Power = 2
    Time = 3

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

   
'''    
Defines request state:
'''
class Request_States:
    In_Queue = 0
    In_System = 1
    Time_Out = 2
    Done = 3    
    
'''
Define cases where state changes can occur:
    N <-> L0 <-> L1 <-> L2 <-> A
'''    
Transitions = np.array([np.array([0, 0, 0, 0, 0]),    # No change
                        np.array([-1, 1, 0, 0, 0]),   # N -> L0
                        np.array([-1, 0, 0, 1, 0]),   # N -> L2 (skip)
                        np.array([1, -1, 0, 0, 0]),   # L0 -> N
                        np.array([0, -1, 1, 0, 0]),   # L0 -> L1
                        np.array([0, -1, 0, 1, 0]),   # L0 -> L2 (skip)
                        np.array([0, 1, -1, 0, 0]),   # L1 -> L0
                        np.array([0, 0, -1, 1, 0]),   # L1 -> L2
                        np.array([0, 0, 1, -1, 0]),   # L2 -> L1
                        ])

State_trans_mapping =np.array([np.array([1,2]),                 # N
                                np.array([3,4,5]),              # L0
                                np.array([6,7]),                # L1
                                np.array([8])],dtype=object)    # L2   

    
'''
Define transition cost for moving to another states:
 state: [RAM, CPU, Power, Time]  
'''    
Transitions_cost = np.array([np.array([2, 0, 0, 0]),                                                                                                                                                            # No change 
                                    np.array([0, 0, 5 * (time := uniform_random_time(3,8)), time]),                                                                                                                   # N -> L0
                                    np.array([1, 0, 5 * (time_1 := uniform_random_time(3,8)) + 40 * ((time_2 := uniform_random_time(5,60)) + (time_3 := uniform_random_time(3,5,step=0.5))), time_1 + time_2 + time_3]),       # N -> L2
                                    np.array([0, 0, 5 * (time := uniform_random_time(1,5)), time]),                                                                                                                   # L0 -> N
                                    np.array([0, 0, 40 * (time := uniform_random_time(5,60)), time]),                                                                                                                 # L0 -> L1
                                    np.array([1, 0, 40 * ((time_1 := uniform_random_time(5,60)) + (time_2 := uniform_random_time(3,5,step=0.5))), time_1 + time_2]),                                                           # L0 -> L2
                                    np.array([0, 0, 5 * (time := uniform_random_time(1,5)), time]),                                                                                                                   # L1 -> L0
                                    np.array([1, 0, 40 * (time := uniform_random_time(3,5,step=0.5)), time]),                                                                                                         # L1 -> L2
                                    np.array([0, 0, 10 * (time := uniform_random_time(30,40)), time])])                                                                                                               # L2 -> L1 


'''
Define resource usage for staying in each state:
 state: [RAM, CPU, Power]
    
'''    
Container_Resource_Usage = np.array([np.array([0, 0, 0]),                     # N
                                            np.array([0, 0, 0]),              # L0
                                            np.array([0, 0, 0]),              # L1
                                            np.array([20 * (time := uniform_random_time(5,60)), cpu_percent := 0.05, cpu_percent * 200]),                        # L2
                                            np.array([20 * (time := uniform_random_time(5,60)), cpu_percent := 0.05 + (0.1 * time), cpu_percent * 200])])        # A


class ServerlessEnv(gym.Env):
    metadata = {}

    def __init__(self, env_config={"render_mode":None, 
                                   "num_service": 1,
                                   "timestep": 120,
                                   "num_container": [100],
                                   "container_lifetime": 3600*8,
                                   "rq_timeout": [20],
                                   "average_requests": 40/60,
                                   "max_rq_active_time": {"type": "random", "value": [60]},
                                   "energy_price": 10e-8, 
                                   "ram_profit": 10e-6,
                                   "cpu_profit": 10e-6,
                                   "alpha": 0.1*0.05,
                                   "beta": 0.1*0.05,
                                   "gamma": 0.1*0.9,    
                                   "log_path": "log.txt"}):
        super(ServerlessEnv, self).__init__()
        '''
        Define environment parameters
        '''
        self.current_time = 0  # Start at time 0
        self.timestep = 120 
        
        self.num_service = env_config["num_service"]  # The number of services
        self.num_ctn_states = len(Container_States.State_Name)
        self.num_trans = Transitions.shape[0] - 1
        
        self.num_container = np.array(env_config["num_container"])
        self.container_lifetime = env_config["container_lifetime"]  # Set lifetime of a container  
        
        self.num_rq_state = len(Container_States.State_Name)  
        self.rq_timeout = env_config["rq_timeout"] 
        self.max_rq_active_time = env_config["max_rq_active_time"]  # "random" or "static"
        self.average_requests = env_config["average_requests"]  # Set the average incoming requests per second 
        self.max_num_request = int(self.average_requests*self.timestep*2)  # Set the limit number of requests that can exist in the system 
        
        self.num_resources = len([attr for attr in vars(Resource_Type) if not attr.startswith('__')]) - 1    # The number of resource parameters (RAM, CPU, Power)
        self.limited_resource = [1000 * 1024, 1000 * 100]  # Set limited amount of [RAM, CPU] of system
        self.energy_price = env_config["energy_price"] # unit cent/Jun/s 
        self.ram_profit = env_config["ram_profit"] # unit cent/Gb/s
        self.cpu_profit = env_config["cpu_profit"] # unit cent/vcpu/s
        self.alpha = env_config["alpha"]
        self.beta = env_config["beta"]
        self.gamma = env_config["gamma"]
        
        '''
        Initialize the state and other variables
        '''

        self.truncated = False
        self.terminated = False
        self.truncated_reason = ""
        self.temp_reward = 0 # Reward for each step
        self.abandone_penalty = 0
        self.delay_penalty = 0
        self.profit = 0
        self.energy_cost = 0
        self.current_resource_usage = np.zeros(self.num_resources,dtype=np.float64)
        self.resource_consumption = np.zeros(self.num_resources,dtype=np.float64)
        

        self._in_queue_requests = [[] for _ in range(self.num_service)] # Requests in queue until current time
        self._in_system_requests = [[] for _ in range(self.num_service)] # Accepted requests in system until current time
        self._done_requests = [[] for _ in range(self.num_service)] # Done requests cache in a timestep
        self._new_requests = [[] for _ in range(self.num_service)] # New incoming requests cache in a timestep
        self._timeout_requests = [[] for _ in range(self.num_service)] # Timeout requests cache in a timestep
        
        self.num_all_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_accepted_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_new_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_in_queue_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_in_sys_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_done_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_rejected_rq = np.zeros(self.num_service,dtype=np.int32)
        self.cu_rq_delay = np.zeros(self.num_service,dtype=np.int32)

        
        self.current_action = 0
        self._action_matrix = np.zeros(shape=(self.num_service,self.num_ctn_states)) 
        self._positive_action_matrix = self._action_matrix * (self._action_matrix > 0)
        self._negative_action_matrix = self._action_matrix * (self._action_matrix < 0)
        self.formatted_action = np.zeros((2,4),dtype=np.int32)
        
        # TODO: try random initial state of container matrix 
        # Create matrix based on self.num_container
        self._container_matrix_tmp = np.hstack((
            self.num_container[:, np.newaxis],  # Convert array to column matrix
            np.zeros((self.num_container.size, self.num_ctn_states-1), dtype=np.int16)  # Matrix of zeros with size 4x3
        )).astype(np.int16)
        # self._container_matrix_tmp = self._create_random_container_matrix()
        self._container_matrix = self._container_matrix_tmp.copy()


        # State space
        self.raw_state_space = self._state_space_init() 
        self.state_space = spaces.flatten_space(self.raw_state_space)
        self.state_size = self.state_space.shape[0]
        
        # State matrices cache
        self._env_matrix = np.zeros((self.num_service, self.num_ctn_states+1),dtype=np.int16)
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix

        # Action space
        self.raw_action_space = self._action_space_init() 
        self.action_size = self._num_action_cal()
        # Only run when initializing the environment
        self.action_space = spaces.Discrete(self.action_size,seed=42)
        
        # Action masking
        self.action_mask = np.zeros((self.action_size),dtype=np.int8)
        self._cal_action_mask()

        assert env_config["render_mode"] is None or env_config["render_mode"] in self.metadata["render_modes"]
        self.render_mode = env_config["render_mode"]
        self.log_file = env_config["log_path"]

     
    # Create action space
    def _action_space_init(self):
        high_matrix = np.zeros((2,self.num_service),dtype=np.int16)
        for service in range(self.num_service):
            high_matrix[0][service]=self.num_container[service]
            high_matrix[1][service]= self.num_trans 
            
        action_space = spaces.Box(low=1,high=high_matrix,shape=(2,self.num_service), dtype=np.int16) # Num container * num transition * num service
        return action_space
    
    # Calculate the number of elements in the action space
    def _num_action_cal(self):
        num_action = 1
        for service  in range(self.num_service): 
            num_action *= (1 + self.num_trans*self.num_container[service])
        return int(num_action)

    # Create state space
    def _state_space_init(self):
        low_matrix = np.zeros((self.num_service, self.num_ctn_states+1),dtype=np.int16)
        high_matrix = np.zeros((self.num_service, self.num_ctn_states+1),dtype=np.int16)
        for service in range(self.num_service):
            for container_state in range(self.num_ctn_states):
                # low_matrix[service][container_state] = -self.num_container[service]
                high_matrix[service][container_state] = 2*self.num_container[service]
            
            # high_matrix[service][Request_States.Done+self.num_ctn_states] = self.max_num_request 
            high_matrix[service][Request_States.In_Queue+self.num_ctn_states] = self.max_num_request 
            # high_matrix[service][Request_States.In_System+self.num_ctn_states] = self.max_num_request 
            # high_matrix[service][Request_States.Time_Out+self.num_ctn_states] = self.max_num_request 
            
        state_space = spaces.Box(low=low_matrix, high=high_matrix, shape=(self.num_service, self.num_ctn_states+1), dtype=np.int16)  # num_service *(num_container_state + num_request_state)
        return state_space
    
    # Calculate the number of elements in the state space
    def _num_state_cal(self):
        ret = 1
        for service in range(self.num_service):
            ret *= compute_formula(self.num_ctn_states,int(self.num_container[service])) 
        ret *= compute_formula(self.num_rq_state,int(2*self.max_num_request))
        return ret

    def _cal_action_mask(self):
        self.action_mask.fill(0)
        self.action_mask[0] = 1
        tmp_action_mask = np.empty((self.num_service),dtype=object) 
        for service in range(self.num_service):
            coefficient = []
            for state in range(self.num_ctn_states-1):
                for trans in State_trans_mapping[state]:
                    coefficient.append({trans:self._container_matrix[service][state]})
            tmp_action_mask[service] = np.array(coefficient)
        
        trans_combs = list(itertools.product(range(1,self.num_trans+1), repeat=self.num_service)) 
        for trans_comb in trans_combs:
            ctn_ranges = []
            for service in range(self.num_service):
                h = tmp_action_mask[service][trans_comb[service]-1][trans_comb[service]]
                ctn_ranges.append(range(1,h + 1))
            for ctn_comb in itertools.product(*ctn_ranges):
                index = self.action_to_number(np.array([list(ctn_comb), list(trans_comb)]))
                self.action_mask[index] = 1
                 
    def _create_random_container_matrix(self):
        ret = np.zeros(shape=(len(self.num_container), self.num_ctn_states),dtype=np.int64)     
        for service in range(self.num_service):
            tmp = self.num_container[service]
            for state in range(self.num_ctn_states-2):
                if tmp > 0 :
                    ret[service][state] = np.random.randint(0,tmp)
                    tmp -= ret[service][state]
            ret[service][self.num_ctn_states-2] = tmp
        return ret
    
    
    def _get_obs(self):
        '''
        Define a function that returns the values of observation
        ''' 
        # Calculate environment matrix
        self._cal_env_matrix()   
        return spaces.flatten(self.raw_state_space,self._env_matrix)


    def _get_reward(self):

        self.temp_reward = self.profit - (self.alpha*self.delay_penalty + self.beta*self.abandone_penalty + self.gamma*self.energy_cost)
        return self.temp_reward
    
    def reset(self, seed=42, options=None):
        '''
        Initialize the environment
        '''
        super().reset(seed=seed) # We need the following line to seed self.np_random
        
        self.current_time = 0  # Start at time 0
        self.current_resource_usage.fill(0)

        # Reset the value of self._container_matrix
        self._container_matrix = self._container_matrix_tmp.copy()
        
        # Observation matrices cache
        self._env_matrix.fill(0)
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix
        
        # self.action_mask.fill(0)
        self._cal_action_mask()
        
        self._in_queue_requests = [[] for _ in range(self.num_service)] 
        self._in_system_requests = [[] for _ in range(self.num_service)] 
        self._done_requests = [[] for _ in range(self.num_service)] 
        self._new_requests = [[] for _ in range(self.num_service)] 
        self._timeout_requests = [[] for _ in range(self.num_service)] 
        
        self.num_all_rq.fill(0)
        self.num_accepted_rq.fill(0)
        self.num_new_rq.fill(0)
        self.num_in_queue_rq.fill(0)
        self.num_in_sys_rq.fill(0)
        self.num_done_rq.fill(0)
        self.num_rejected_rq.fill(0)
        self.cu_rq_delay.fill(0)
        self.resource_consumption.fill(0)
    
        self.truncated = False
        self.terminated = False
        
        observation = self._get_obs()
        
        return observation
     
           
    def _receive_new_requests(self):
        num_new_rq = rq.generate_requests(self._in_queue_requests,
                                           size=self.num_service,
                                           current_time=self.current_time, 
                                           avg_requests_per_second=self.average_requests,
                                           timeout=self.rq_timeout,
                                           max_rq_active_time=self.max_rq_active_time)
        self.num_all_rq += num_new_rq
        self.num_new_rq += num_new_rq
            
    def _set_truncated(self):
        temp = self._container_matrix + self._action_matrix
        
        # temp_current_usage = np.sum(np.dot(self._container_matrix, Container_Resource_Usage),axis=0)
        # # Instantaneous resource consumption due to state transition
        # if (temp_current_usage[Resource_Type.CPU] > self.limited_resource[Resource_Type.CPU]
        #     or temp_current_usage[Resource_Type.RAM] > self.limited_resource[Resource_Type.RAM]):
        #     # If instantaneous resource consumption exceeds the limit, state transition is not allowed
        #     self._action_matrix.fill(0)
        # else: 
        #     pass

        if (np.any(temp < 0)):
            # If the number of containers is less than 0, state transition is not allowed
            self.truncated = True
            self.truncated_reason = "Wrong number action"
            print(self.truncated_reason)
            print(self._container_matrix)
            print(self._action_matrix)
            print(self.current_action)
            print(self.action_mask[self.current_action])
            print(self.number_to_action(self.current_action))
            print(self.current_time)
        else: 
            pass
            
        
            
    def _set_terminated(self):
        if (self.current_time >= self.container_lifetime):
            self.terminated = True                      
            
    def _handle_env_change(self):
        self._positive_action_matrix = self._action_matrix * (self._action_matrix > 0)
        self._negative_action_matrix = self._action_matrix * (self._action_matrix < 0)
        
        self._container_matrix += self._negative_action_matrix
        relative_time = 0
        while relative_time < self.timestep:
            self._receive_new_requests()
            self.current_resource_usage = np.sum(np.dot(self._container_matrix,Container_Resource_Usage),axis=0)
            for service in range(self.num_service):
                # State transition of container 
                trans_num  = self.formatted_action[0][service]
                trans_type = self.formatted_action[1][service]
                
                if relative_time == Transitions_cost[trans_type][Resource_Type.Time]:
                    self._container_matrix[service] += self._positive_action_matrix[service]
                elif relative_time < Transitions_cost[trans_type][Resource_Type.Time]:
                    # Instantaneous resource consumption due to state transition   
                    self.current_resource_usage[Resource_Type.CPU] += Transitions_cost[trans_type][Resource_Type.CPU]*trans_num 
                    self.current_resource_usage[Resource_Type.RAM] += Transitions_cost[trans_type][Resource_Type.RAM]*trans_num 
                    self.current_resource_usage[Resource_Type.Power] += Transitions_cost[trans_type][Resource_Type.Power]*trans_num
                
                # Handle requests in queue
                for rq in self._in_queue_requests[service][:]:
                    # Release requests that have timed out
                    if self.current_time == rq.time_out + rq.in_queue_time:
                        rq.set_state(Request_States.Time_Out)
                        rq.set_out_system_time(self.current_time)
                        self._timeout_requests[service].append(rq)
                        self._in_queue_requests[service].remove(rq)
                    else:
                        # If there are available resources, push the request into the system
                        if self._container_matrix[service][Container_States.Warm_CPU] > 0:
                            rq.set_state(Request_States.In_System)
                            rq.set_in_system_time(self.current_time)
                            self.num_accepted_rq[service] += 1
                            self._in_system_requests[service].append(rq)
                            self._in_queue_requests[service].remove(rq)
                            self._container_matrix[service][Container_States.Active] += 1
                            self._container_matrix[service][Container_States.Warm_CPU] -= 1
                            
                            # Delay penalty is applied only once at the time the request is accepted by the system
                            delay_time = rq.in_system_time - rq.in_queue_time
                            self.delay_penalty += Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*delay_time
                            self.delay_penalty += Request_Resource_Usage[service][Resource_Type.CPU]*self.cpu_profit*delay_time
                            
                            self.cu_rq_delay[service] += delay_time

                # Handle requests in system
                for rq in self._in_system_requests[service][:]:
                    # Resource consumption by request
                    self.current_resource_usage += Request_Resource_Usage[service]
                    # Release requests that have been completed
                    if rq.active_time == self.current_time - rq.in_system_time:
                        rq.set_state(Request_States.Done)
                        rq.set_out_system_time(self.current_time)
                        self._done_requests[service].append(rq)
                        self._in_system_requests[service].remove(rq)
                        self._container_matrix[service][Container_States.Active] -= 1
                        self._container_matrix[service][Container_States.Warm_CPU] += 1
                        
                        # Abandon penalty is applied only once at the time the request times out and is rejected by the system
                        in_queue_time = rq.out_system_time - rq.in_queue_time
                        self.abandone_penalty += Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*in_queue_time
                        self.abandone_penalty += Request_Resource_Usage[service][Resource_Type.CPU]*self.ram_profit*in_queue_time
                
                # Profit of requests accepted into the system in 1 second
                self.profit += Request_Resource_Usage[service][Resource_Type.RAM]*self.ram_profit*self._container_matrix[service][Container_States.Active]
                self.profit += Request_Resource_Usage[service][Resource_Type.CPU]*self.cpu_profit*self._container_matrix[service][Container_States.Active]
            
            self.energy_cost += self.current_resource_usage[Resource_Type.Power]*self.energy_price 
            self.resource_consumption += self.current_resource_usage
                
            self.current_time += 1
            relative_time += 1
        

    
    def _cal_system_evaluation(self):
        for service in range(self.num_service):    
            self.num_in_queue_rq[service] = len(self._in_queue_requests[service])
            self.num_in_sys_rq[service] = self._container_matrix[service][Container_States.Active]
            self.num_done_rq[service] = len(self._done_requests[service])
            self.num_rejected_rq[service] = len(self._timeout_requests[service])
            
    def  _cal_env_matrix(self):
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix
        for service in range(self.num_service):
            self._env_matrix[service][self.num_ctn_states+Request_States.In_Queue] = len(self._in_queue_requests[service])
  
    
    def _action_to_matrix(self,index):
        self.current_action = index
        action = self.number_to_action(index)
        action = action.reshape(2,self.num_service)
        action_coefficient = np.diag(action[0])
        action_unit = []
        for service in action[1]:
            action_unit.append(Transitions[service])
        self._action_matrix = action_coefficient @ action_unit
        return action
        
    def _clear_cache(self):
        self._new_requests = [[] for _ in range(self.num_service)] 
        self._done_requests = [[] for _ in range(self.num_service)] 
        self._timeout_requests = [[] for _ in range(self.num_service)] 
        self._env_matrix.fill(0)
        self.action_mask.fill(0)
        self.temp_reward = 0
        self.abandone_penalty = 0
        self.delay_penalty = 0
        self.profit = 0
        self.energy_cost = 0
        self.num_new_rq.fill(0)
        self.num_in_queue_rq.fill(0)
        self.num_in_sys_rq.fill(0)
        self.num_done_rq.fill(0)
        self.num_rejected_rq.fill(0)
        self.resource_consumption.fill(0)
        self.truncated = False
        self.terminated = False
                 
    def _pre_step(self,action):
        self._clear_cache()
        self.formatted_action = self._action_to_matrix(action)
        self._set_terminated()
        self._set_truncated()
        
        
    def step(self, action):
        self._pre_step(action)
        self._handle_env_change()   
        self._cal_system_evaluation()
        observation = self._get_obs()
        reward = self._get_reward()
        self._cal_action_mask()
        
        return observation, reward, self.terminated, self.truncated
    
    def render(self):
        '''
        Implement a visualization method
        '''

        with open(self.log_file, 'a') as f:
            f.write("-------------------------------------------------------------\n")
            f.write("SYSTEM EVALUATION PARAMETERS IN TIMESTEP {}: \n".format(self.current_time // self.timestep))
            f.write("- Action number: {}\n".format(self.current_action))
            f.write("- Action matrix: \n{}\n".format(self._action_matrix))
            f.write("- Containers state after action: \n{}\n".format(self._container_matrix))
            f.write("- Cumulative number request : {}\n".format(self.num_all_rq))
            f.write("- Cumulative number accpeted request : {}\n".format(self.num_accepted_rq))
            f.write("- Number new request: {}\n".format(self.num_new_rq))
            f.write("- Number in queue request: {}\n".format(self.num_in_queue_rq))
            f.write("- Number in system request: {}\n".format(self.num_in_sys_rq))
            f.write("- Number done system request: {}\n".format(self.num_done_rq))
            f.write("- Number timeout system request: {}\n".format(self.num_rejected_rq))
            f.write("- Cumulative request delay : {}\n".format(self.cu_rq_delay))
            f.write("- Rewards: {:.4f} = ".format(self.temp_reward))
            f.write("(Profit: {:.4f}) - ".format(self.profit))
            f.write("{:.4f}*(Abandone penalty: {:.4f}) - ".format(self.alpha, self.abandone_penalty))
            f.write("{:.4f}*(Delay penalty: {:.4f}) - ".format(self.beta, self.delay_penalty))
            f.write("{:.4f}*(Energy cost: {:.4f})\n".format(self.gamma, self.energy_cost))
            f.write("Energy consumption over timestep : {:.4f}J, ".format(self.resource_consumption[Resource_Type.Power]))
            f.write("RAM consumption over timestep  : {:.4f}Gb, ".format(self.resource_consumption[Resource_Type.RAM]))
            f.write("CPU consumption over timestep  : {:.4f}Core \n".format(self.resource_consumption[Resource_Type.CPU]))
            f.write("Current energy usage: {:.4f}J, ".format(self.current_resource_usage[Resource_Type.Power]))
            f.write("Current RAM usage: {:.4f}Gb, ".format(self.current_resource_usage[Resource_Type.RAM]))
            f.write("Current CPU usage: {:.4f}Core \n".format(self.current_resource_usage[Resource_Type.CPU]))
            
    def action_to_number(self, action_matrix):
        index = 0
        multiplier = 1
        for service in range(self.num_service):
            if action_matrix[0][service] == 0:
                index += 0
            else:
                index += multiplier*(action_matrix[0][service] + (action_matrix[1][service]-1)*self.num_container[service])
            multiplier *= (self.num_container[service]*self.num_trans + 1)
        return int(index)

    def number_to_action(self, index):
        result = np.zeros((2,self.num_service),dtype=np.int32)
        tmp = 0
        multiplier = 1 
        for service in range(self.num_service-1):
            multiplier *= (self.num_container[service]*self.num_trans + 1)
            
        for service in reversed(range(self.num_service)):
            tmp = index // multiplier 
            if tmp == 0:
                result[0][service] = 0
                result[1][service] = 0
            else:
                result[0][service] = ((tmp-1) % self.num_container[service]) + 1
                result[1][service] = ((tmp-1) // self.num_container[service]) + 1
            index %= multiplier
            multiplier //= (self.num_container[service-1]*self.num_trans + 1)
        
        return result


if __name__ == "__main__":
    # Create the serverless environment
    env = ServerlessEnv()
    print(env._container_matrix)
    # Reset the environment to the initial state
    observation = env.reset()
    # Perform random actions
    i = 0
    while (i<10000):
        # env._cal_action_mask()
        action = env.action_space.sample(mask=env.action_mask)  # Random action
        observation, reward, terminated, truncated = env.step(action)
        env.render()
        i += 1
        if truncated:
            print("error")
            break
        if (terminated): 
            print("--------------------------------cff--------")
            break
            env.reset()
        else: continue