import numpy as np
import gymnasium as gym
import rlss_env.request as rq
import itertools
import math

from rlss_env.request import Request_States
import rlss_env.profiling as profiling
from rlss_env.profiling import Resource_Type
from rlss_env.container import Container_States

def compute_formula(num_box, num_ball):
    numerator = math.factorial(num_box + num_ball - 1)
    denominator = math.factorial(num_box - 1) * math.factorial(num_ball)
    return int(numerator/denominator)

class ServerlessEnv(gym.Env):
    metadata = {}
    def __init__(self, traffic_generator, render_mode=None, num_service=1, timestep=120, 
                 num_container=[100], container_lifetime=3600*8, energy_price=8e-6, 
                 ram_profit=10e-6, cpu_profit=10e-6, delay_coff=1, aban_coff=1, energy_coff=1, 
                 custom_profiling=False, profiling_path=None, queue_size=5000):
        # Initialize other attributes and perform setup as needed
        super(ServerlessEnv, self).__init__() 
         
        self.TRANS = np.array([np.array([0, 0, 0, 0, 0]),    # No change
                               np.array([-1, 1, 0, 0, 0]),   # N -> L0
                               np.array([-1, 0, 0, 1, 0]),   # N -> L2 (skip)
                               np.array([1, -1, 0, 0, 0]),   # L0 -> N
                               np.array([0, -1, 1, 0, 0]),   # L0 -> L1
                               np.array([0, -1, 0, 1, 0]),   # L0 -> L2 (skip)
                               np.array([0, 1, -1, 0, 0]),   # L1 -> L0
                               np.array([0, 0, -1, 1, 0]),   # L1 -> L2
                               np.array([0, 0, 1, -1, 0]),   # L2 -> L1
                               ])

        self.STATE_TRANS_MAPPING =np.array([np.array([1,2]),                  # N
                                            np.array([3,4,5]),                # L0
                                            np.array([6,7]),                  # L1
                                            np.array([8])],dtype=object)      # L2   
        
        self.cont_res_usage = None
        self.trans_cost = None
        self.get_resource_profiling(profiling_path, custom_profiling)
        
        self.current_time = 0  # Start at time 0
        self.timestep = timestep
        
        self.num_service = num_service  # The number of services
        self.num_ctn_states = len(Container_States.State_Name)
        self.num_trans = self.TRANS.shape[0] - 1
        
        self.num_container = np.array(num_container)
        self.container_lifetime = container_lifetime  # Set lifetime of a container  
        
        self.num_rq_state = len(Container_States.State_Name)  
    
    
        self.traffic_generator = traffic_generator
        self.queue_size = queue_size
        
        self.num_resources = len([attr for attr in vars(Resource_Type) if not attr.startswith('__')]) - 1   
        self.limited_resource = [1000 * 1024, 1000 * 100]  # Set limited amount of [RAM, CPU] of system
        self.energy_price = energy_price # unit cent/Jun/s 
        self.ram_profit = ram_profit # unit cent/Gb/s
        self.cpu_profit = cpu_profit # unit cent/vcpu/s
        self.delay_coff = delay_coff
        self.aban_coff = aban_coff
        self.engery_coff = energy_coff
        
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
        self.per_second_resource_usage = np.zeros(self.num_resources,dtype=np.float64)
        self.cum_resource_usage = np.zeros(self.num_resources,dtype=np.float64)
        
        # TODO: change queue have a size
        self._in_queue_requests = [[] for _ in range(self.num_service)] 
        self._in_system_requests = [[] for _ in range(self.num_service)] 
        self._done_requests = [[] for _ in range(self.num_service)]
        self._new_requests = [[] for _ in range(self.num_service)] 
        self._timeout_requests = [[] for _ in range(self.num_service)] 
        
        self.num_accepted_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_new_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_in_queue_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_in_sys_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_done_rq = np.zeros(self.num_service,dtype=np.int32)
        self.num_rejected_rq = np.zeros(self.num_service,dtype=np.int32)
        self.rq_delay = [[] for _ in range(self.num_service)] 
        

        
        self.current_action = 0
        self._action_matrix = np.zeros(shape=(self.num_service,self.num_ctn_states)) 
        self._positive_action_matrix = self._action_matrix * (self._action_matrix > 0)
        self._negative_action_matrix = self._action_matrix * (self._action_matrix < 0)
        self.formatted_action = np.zeros((2,4),dtype=np.int32)
        
        # Create matrix based on self.num_container
        self._container_matrix_tmp = np.hstack((
            self.num_container[:, np.newaxis],  
            np.zeros((self.num_container.size, self.num_ctn_states-1), dtype=np.int16)
        )).astype(np.int16)
        # self._container_matrix_tmp = self._create_random_container_matrix()
        self._container_matrix = self._container_matrix_tmp.copy()


        # State space
        self.raw_state_space = self._state_space_init() 
        self.state_space = gym.spaces.flatten_space(self.raw_state_space)
        self.state_size = self.state_space.shape[0]
        
        # State matrices cache
        self._env_matrix = np.zeros((self.num_service, self.num_ctn_states+1),dtype=np.int16)
        self._env_matrix[:,0:self.num_ctn_states]=self._container_matrix

        # Action space
        self.raw_action_space = self._action_space_init() 
        self.action_size = self._num_action_cal()
        # Only run when initializing the environment
        self.action_space = gym.spaces.Discrete(self.action_size,seed=42)
        
        # Action masking
        self.action_mask = np.zeros((self.action_size),dtype=np.int8)
        self._cal_action_mask()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

     
    # Create action space
    def _action_space_init(self):
        high_matrix = np.zeros((2,self.num_service),dtype=np.int16)
        for service in range(self.num_service):
            high_matrix[0][service]=self.num_container[service]
            high_matrix[1][service]= self.num_trans 
            
        action_space = gym.spaces.Box(low=1,high=high_matrix,shape=(2,self.num_service), dtype=np.int16) # Num container * num transition * num service
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
            high_matrix[service][Request_States.In_Queue+self.num_ctn_states] = self.queue_size 
            # high_matrix[service][Request_States.In_System+self.num_ctn_states] = self.max_num_request 
            # high_matrix[service][Request_States.Time_Out+self.num_ctn_states] = self.max_num_request 
            
        state_space = gym.spaces.Box(low=low_matrix, high=high_matrix, shape=(self.num_service, self.num_ctn_states+1), dtype=np.int16) 
        return state_space
    
    # Calculate the number of elements in the state space
    def _num_state_cal(self):
        ret = 1
        for service in range(self.num_service):
            ret *= compute_formula(self.num_ctn_states,int(self.num_container[service])) 
        ret *= compute_formula(self.num_rq_state,int(2*self.queue_size))
        return ret

    def _cal_action_mask(self):
        self.action_mask.fill(0)
        self.action_mask[0] = 1
        tmp_action_mask = np.empty((self.num_service),dtype=object) 
        for service in range(self.num_service):
            coefficient = []
            for state in range(self.num_ctn_states-1):
                for trans in self.STATE_TRANS_MAPPING[state]:
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
    

    def get_resource_profiling(self, profiling_path, custom_profiling):
        if  custom_profiling is False:
            self.cont_res_usage = profiling.generate_container_resource_usage()
            self.trans_cost = profiling.generate_trans_cost()
            if profiling_path:
                np.savez(profiling_path, trans_cost=self.trans_cost, cont_res_usage=self.cont_res_usage)
            else:
                np.savez("profiling.npz", trans_cost=self.trans_cost, cont_res_usage=self.cont_res_usage)      
        else:
            if profiling_path:
                data = np.load(profiling_path)
            else:
                data = np.load("profiling.npz")
                
            self.cont_res_usage = data["cont_res_usage"]
            self.trans_cost = data["trans_cost"]


    def _get_obs(self):
        '''
        Define a function that returns the values of observation
        ''' 
        # Calculate environment matrix
        self._cal_env_matrix()   
        return gym.spaces.flatten(self.raw_state_space,self._env_matrix)


    def _get_reward(self):
        self.temp_reward = 2 + self.profit - (self.delay_coff*self.delay_penalty + self.aban_coff*self.abandone_penalty + self.engery_coff*self.energy_cost)
        return self.temp_reward
    
    def reset(self, seed=42, options=None):
        '''
        Initialize the environment
        '''
        super().reset(seed=seed) # We need the following line to seed self.np_random
        
        self.current_time = 0  # Start at time 0
        self.per_second_resource_usage.fill(0)

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
        
        self.num_accepted_rq.fill(0)
        self.num_new_rq.fill(0)
        self.num_in_queue_rq.fill(0)
        self.num_in_sys_rq.fill(0)
        self.num_done_rq.fill(0)
        self.num_rejected_rq.fill(0)
        self.cum_resource_usage.fill(0)
        
        self.cu_rq_delay = [[] for _ in range(self.num_service)]
    
        self.truncated = False
        self.terminated = False
        
        observation = self._get_obs()
        
        return observation
     
           
    def _receive_new_requests(self):
        num_new_rq = self.traffic_generator.generate_requests(self._in_queue_requests,
                                                              current_time=self.current_time)
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
            self.per_second_resource_usage = np.sum(np.dot(self._container_matrix,self.cont_res_usage),axis=0)
            for service in range(self.num_service):
                # State transition of container 
                trans_num  = self.formatted_action[0][service]
                trans_type = self.formatted_action[1][service]
                
                if relative_time == np.ceil(self.trans_cost[trans_type][Resource_Type.Time]):
                    self._container_matrix[service] += self._positive_action_matrix[service]
                elif relative_time < np.ceil(self.trans_cost[trans_type][Resource_Type.Time]):
                    # Instantaneous resource consumption due to state transition   
                    self.per_second_resource_usage[Resource_Type.CPU] += self.trans_cost[trans_type][Resource_Type.CPU]*trans_num 
                    self.per_second_resource_usage[Resource_Type.RAM] += self.trans_cost[trans_type][Resource_Type.RAM]*trans_num 
                    self.per_second_resource_usage[Resource_Type.Power] += self.trans_cost[trans_type][Resource_Type.Power]*trans_num
                
                # Handle requests in queue
                for rq in self._in_queue_requests[service][:]:
                    # Release requests that have timed out
                    if self.current_time == np.ceil(rq.time_out + rq.in_queue_time):
                        rq.set_state(Request_States.Time_Out)
                        rq.set_out_system_time(self.current_time)
                        self._timeout_requests[service].append(rq)
                        self._in_queue_requests[service].remove(rq)
                        
                        # Abandon penalty is applied only once at the time the request times out and is rejected by the system
                        self.abandone_penalty += self.cont_res_usage[Container_States.Active][Resource_Type.RAM]*self.ram_profit*rq.active_time
                        self.abandone_penalty += self.cont_res_usage[Container_States.Active][Resource_Type.CPU]*self.cpu_profit*rq.active_time
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
                            self.delay_penalty += self.cont_res_usage[Container_States.Active][Resource_Type.RAM]*self.ram_profit*delay_time
                            self.delay_penalty += self.cont_res_usage[Container_States.Active][Resource_Type.CPU]*self.cpu_profit*delay_time
                            
                            self.rq_delay[service].append(delay_time)

                # Handle requests in system
                for rq in self._in_system_requests[service][:]:
                    # Resource consumption by request
                    # self.per_second_resource_usage += REQ_RES_USAGE[service]
                    # Release requests that have been completed
                    if rq.active_time == np.ceil(self.current_time - rq.in_system_time):
                        rq.set_state(Request_States.Done)
                        rq.set_out_system_time(self.current_time)
                        self._done_requests[service].append(rq)
                        self._in_system_requests[service].remove(rq)
                        self._container_matrix[service][Container_States.Active] -= 1
                        self._container_matrix[service][Container_States.Warm_CPU] += 1
                
                # Profit of requests accepted into the system in 1 second
                self.profit += self.cont_res_usage[Container_States.Active][Resource_Type.RAM]*self.ram_profit*self._container_matrix[service][Container_States.Active]
                self.profit += self.cont_res_usage[Container_States.Active][Resource_Type.CPU]*self.cpu_profit*self._container_matrix[service][Container_States.Active]
            
            self.energy_cost += self.per_second_resource_usage[Resource_Type.Power]*self.energy_price 
            self.cum_resource_usage += self.per_second_resource_usage
                
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
            action_unit.append(self.TRANS[service])
        self._action_matrix = action_coefficient @ action_unit
        return action
        
    def _clear_cache(self):
        self._new_requests = [[] for _ in range(self.num_service)] 
        self._done_requests = [[] for _ in range(self.num_service)] 
        self._timeout_requests = [[] for _ in range(self.num_service)] 
        self.rq_delay = [[] for _ in range(self.num_service)] 
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
        self.num_accepted_rq.fill(0)
        self.num_rejected_rq.fill(0)
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
        Implement a visualization method that saves logs in dictionary format
        '''
        log_data = {
            "Episode": None,
            "timestep": self.current_time // self.timestep,
            "action_number": self.current_action,
            "action_matrix": self._action_matrix.tolist(),  
            "containers_state_after_action": self._container_matrix.tolist(),
            "cumulative_number_accepted_request": self.num_accepted_rq.tolist(),
            "new_requests": self.num_new_rq.tolist(),
            "queue_requests": self.num_in_queue_rq.tolist(),
            "system_requests": self.num_in_sys_rq.tolist(),
            "done_requests": self.num_done_rq.tolist(),
            "timeout_requests": self.num_rejected_rq.tolist(),
            "request_delay": self.rq_delay,
            "step_reward": self.temp_reward,
            "profit": self.profit,
            "abandone_penalty value": self.aban_coff*self.abandone_penalty,
            "delay_penalty value": self.delay_coff*self.delay_penalty,
            "energy_cost value": self.engery_coff*self.energy_cost,
            "energy_consumption": self.cum_resource_usage[Resource_Type.Power],
            "ram_consumption": self.cum_resource_usage[Resource_Type.RAM],
            "cpu_consumption": self.cum_resource_usage[Resource_Type.CPU],
            "per_second_energy_usage": self.per_second_resource_usage[Resource_Type.Power],
            "per_second_ram_usage": self.per_second_resource_usage[Resource_Type.RAM],
            "per_second_cpu_usage": self.per_second_resource_usage[Resource_Type.CPU]
        } 
        return log_data
            
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
