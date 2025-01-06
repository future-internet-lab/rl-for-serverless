import numpy as np
import gymnasium as gym
import rlss_env.request as req
import math

from rlss_env.request import Request_States
import rlss_env.profiling as profiling
from rlss_env.profiling import Resource_Type
from rlss_env.container import Container_States

class ServerlessEnv(gym.Env):
    metadata = {}
    def __init__(self, traffic_generator, render_mode=None, num_service=1, step_interval=120, 
                 num_container=[100], cont_span=3600*8, energy_price=8e-6, 
                 ram_profit=10e-6, cpu_profit=10e-6, delay_coff=1, aban_coff=1, energy_coff=1, 
                 custom_profiling=False, profiling_path=None, queue_size=5000, reward_add=0):
        """
        Initialize the ServerlessEnv environment.
        Parameters:
        traffic_generator (object): The traffic generator.
        render_mode (str, optional): The render mode. Defaults to None.
        num_service (int, optional): The number of services. Defaults to 1.
        step_interval (int, optional): The step_interval of time between two steps. Defaults to 120.
        num_container (list, optional): The maximum number of containers for each service. Defaults to [100].
        cont_span (int, optional): The time span of the container. Defaults to 3600*8.
        energy_price (float, optional): The unit energy price in cent/Jun/s. Defaults to 8e-6.
        ram_profit (float, optional): The unit RAM profit in cent/Gb/s. Defaults to 10e-6.
        cpu_profit (float, optional): The unit CPU profit in cent/vcpu/s. Defaults to 10e-6.
        delay_coff (float, optional): The delay penalty coefficient. Defaults to 1.
        aban_coff (float, optional): The abandon penalty coefficient. Defaults to 1.
        energy_coff (float, optional): The energy cost coefficient. Defaults to 1.
        custom_profiling (bool, optional): Whether to use custom profiling. Defaults to False.
        profiling_path (str, optional): The path to the profiling file. Defaults to None.
        queue_size (int, optional): The size of the queue. Defaults to 5000.
        reward_add (int, optional): Additional to make the reward positive. Defaults to 0.
        """
        super(ServerlessEnv, self).__init__() 
         
        self.TRANS = np.array([np.array([0, 0, 0, 0, 0]),    # No change
                               np.array([-1, 1, 0, 0, 0]),   # Null -> Cold
                               np.array([-1, 0, 0, 1, 0]),   # Null -> Warm_cpu (skip)
                               np.array([1, -1, 0, 0, 0]),   # Cold -> Null
                               np.array([0, -1, 1, 0, 0]),   # Cold -> Warm_disk
                               np.array([0, -1, 0, 1, 0]),   # Cold -> Warm_cpu (skip)
                               np.array([0, 1, -1, 0, 0]),   # Warm_disk -> Cold
                               np.array([0, 0, -1, 1, 0]),   # Warm_disk -> Warm_cpu 
                               np.array([0, 0, 1, -1, 0]),   # Warm_cpu -> Warm_disk
                               ])

        self.TRANS_ST_MAPPING = {1: Container_States.Null,
                                 2: Container_States.Null,
                                 3: Container_States.Cold,
                                 4: Container_States.Cold,
                                 5: Container_States.Cold,
                                 6: Container_States.Warm_Disk,
                                 7: Container_States.Warm_Disk,
                                 8: Container_States.Warm_CPU,
                                 }
        
        self.cont_res_usage = None # Container resource usage
        self.trans_cost = None # Transition cost
        self.get_res_profile(profiling_path, custom_profiling)
        
        self.now = 0  # Start at time 0
        self.step_interval = step_interval # The step_interval of time between two steps
        
        self.n_svc = num_service  # The number of services
        self.n_ctn_states = len(Container_States.State_Name) # The number of container states
        self.n_trans = self.TRANS.shape[0] - 1 # The number of transitions, excluding the no-change transition
        
        self.max_n_ctn = np.array(num_container) # The maximum number of containers for each service
        self.cont_span = cont_span  # The time span of the container
        
        self.n_req_states = len(Container_States.State_Name)  # The number of request states
    
        self.traffic_generator = traffic_generator # The traffic generator
        self.queue_size = queue_size # The size of the queue (currently not used)
        
        self.n_res_types = len([attr for attr in vars(Resource_Type) if not attr.startswith('__')]) - 1   
        self.res_limit = [1000 * 1024, 1000 * 100]  # Set [RAM, CPU] limit of system
        self.en_price = energy_price # unit cent/Jun/s 
        self.ram_profit = ram_profit # unit cent/Gb/s
        self.cpu_profit = cpu_profit # unit cent/vcpu/s
        self.delay_coff = delay_coff # Delay penalty coefficient
        self.aban_coff = aban_coff # Abandon penalty coefficient
        self.en_coff = energy_coff # Energy cost coefficient
        
        '''
        Initialize the state and other variables
        '''

        self.truncated = False # Whether the environment is truncated
        self.terminated = False # Whether the environment is terminated
        self.truncated_reason = "" # The reason for truncation
        self.step_rwd = 0 # Reward for each step
        self.aban_pen = 0 # Abandon penalty in current step
        self.delay_pen = 0 # Delay penalty in current step
        self.profit = 0 # Profit in current step
        self.en_cost = 0 # Energy cost in current step
        self.ptu_res_usage  = np.zeros(self.n_res_types,dtype=np.float64) # Resource usage in current time unit
        self.cum_res_usage = np.zeros(self.n_res_types,dtype=np.float64) # Cumulative resource usage
        
        # TODO: change queue have a size
        self.queued_reqs = [[] for _ in range(self.n_svc)] # Queued requests
        self.active_reqs = [[] for _ in range(self.n_svc)] # Active requests
        self.done_reqs = [[] for _ in range(self.n_svc)] # Done requests in current step
        self.new_reqs = [[] for _ in range(self.n_svc)] # New requests in current step
        self.rej_reqs = [[] for _ in range(self.n_svc)] # Rejected requests in current step
        
        self.total_accepted_req  = np.zeros(self.n_svc,dtype=np.uint32) # Total number of accepted requests in current step
        self.total_new_req = np.zeros(self.n_svc,dtype=np.uint32) # Total number of new requests in current step
        self.total_done_req = np.zeros(self.n_svc,dtype=np.uint32) # Total number of done requests in current step
        self.total_rej_req = np.zeros(self.n_svc,dtype=np.uint32) # Total number of rejected requests in current step
        
        self.cur_n_queued_req = np.zeros(self.n_svc,dtype=np.uint32) # Current number of queued requests
        self.cur_n_active_req = np.zeros(self.n_svc,dtype=np.uint32) # Current number of active requests
        self.req_delay = [[] for _ in range(self.n_svc)] # Delay of requests in current step
        

        
        self.cur_act_idx = 0 # Current action index
        self.cur_act_mtx = np.zeros(shape=(self.n_svc,self.n_ctn_states))
        # self.pos_act_mtx = self.cur_act_mtx * (self.cur_act_mtx > 0)
        # self.neg_act_mtx = self.cur_act_mtx * (self.cur_act_mtx < 0)
        self.fmt_act = np.zeros((2,4),dtype=np.int32)
        
        # Create matrix based on self.max_n_ctn
        self._cont_st_mtx_init = np.hstack((
            self.max_n_ctn[:, np.newaxis],  
            np.zeros((self.max_n_ctn.size, self.n_ctn_states-1), dtype=np.int16)
        )).astype(np.int16)
        # self._cont_st_mtx_init = self.gen_rand_cont_mtx()
        self._cont_st_mtx = self._cont_st_mtx_init.copy()


        # State space
        self.raw_state_space = self.state_space_init() 
        self.state_space = gym.spaces.flatten_space(self.raw_state_space)
        self.state_size = self.state_space.shape[0]
        
        # observation matrix: we use observation and state interchangeably
        # https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#states-and-observations
        self.obs_mtx = np.zeros((self.n_svc, self.n_ctn_states+1),dtype=np.int16)
        self.obs_mtx[:,0:self.n_ctn_states] = self._cont_st_mtx

        # Our action space is a Box space with a shape of (2, self.n_svc) with diffirent limit for each service column
        self.raw_action_space = self.action_space_init() 
        # Our action space too complex to use gym's flatten_space, so we implement our one.
        self.action_size = self._num_action_cal() 
        self.action_space = gym.spaces.Discrete(self.action_size,seed=42)
        
        # Action masking
        self.action_mask = np.zeros((self.action_size),dtype=np.int8)
        self._cal_action_mask()
        
        self.rwd_add = reward_add

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

     
    def action_space_init(self):
        """
        Initializes the action space for the environment.
        The action space is defined as a Box space with a shape of (2, self.n_svc).
        The high values for the Box space are determined by the maximum number of containers
        and the number of transitions for each service.
        Returns:
            gym.spaces.Box: The initialized action space.
        """
        high_matrix = np.zeros((2,self.n_svc),dtype=np.int16)
        for svc in range(self.n_svc):
            high_matrix[0][svc]=self.max_n_ctn[svc]
            high_matrix[1][svc]= self.n_trans 
            
        # Num container * num transition * num service
        action_space = gym.spaces.Box(low=1,high=high_matrix,shape=(2,self.n_svc), dtype=np.int16)
        return action_space
    
    
    def state_space_init(self):
        """
        Initializes the state space for the environment.
        This function creates a state space represented by a matrix with dimensions 
        (self.n_svc, self.n_ctn_states + 1). The state space is defined using the 
        gym.spaces.Box class, with the lower and upper bounds specified by low_matrix 
        and high_matrix respectively.
        The low_matrix is initialized to zeros, while the high_matrix is initialized 
        based on the maximum number of containers (self.max_n_ctn) and the queue size 
        (self.queue_size) for each service (svc).
        Returns:
            gym.spaces.Box: The initialized state space with specified bounds and shape.
        """
        low_matrix = np.zeros((self.n_svc, self.n_ctn_states+1),dtype=np.int16)
        high_matrix = np.zeros((self.n_svc, self.n_ctn_states+1),dtype=np.int16)
        for svc in range(self.n_svc):
            for container_state in range(self.n_ctn_states):
                # low_matrix[svc][container_state] = -self.max_n_ctn[svc]
                high_matrix[svc][container_state] = 2*self.max_n_ctn[svc]
            
            # high_matrix[svc][Request_States.Done+self.n_ctn_states] = self.max_num_request 
            high_matrix[svc][Request_States.In_Queue+self.n_ctn_states] = self.queue_size 
            # high_matrix[svc][Request_States.In_System+self.n_ctn_states] = self.max_num_request 
            # high_matrix[svc][Request_States.Rejected+self.n_ctn_states] = self.max_num_request 
            
        state_space = gym.spaces.Box(low=low_matrix, high=high_matrix, shape=(self.n_svc, self.n_ctn_states+1), dtype=np.int16) 
        return state_space


    def _num_action_cal(self):
        """
        Calculate the total number of possible actions.

        This method computes the total number of possible actions based on the number of services (`n_svc`), 
        the number of transitions (`n_trans`), and the maximum number of containers per service (`max_n_ctn`).

        Returns:
            int: The total number of possible actions.
        """
        num_action = 1
        for svc  in range(self.n_svc): 
            num_action *= (1 + self.n_trans*self.max_n_ctn[svc])
        return int(num_action)

    
    def _num_state_cal(self):
        """
        Calculate the number of possible states in the environment.
        This method computes the total number of possible states by multiplying the 
        results of the `compute_formula` function for each service and request state.
        The `compute_formula` function calculates the binomial coefficient, which is 
        used to determine the number of ways to distribute a given number of balls 
        into a given number of boxes.
        Returns:
            int: The total number of possible states in the environment.
        """
        def compute_formula(num_box, num_ball):
            numerator = math.factorial(num_box + num_ball - 1)
            denominator = math.factorial(num_box - 1) * math.factorial(num_ball)
            return int(numerator/denominator)
        
        ret = 1
        for svc in range(self.n_svc):
            ret *= compute_formula(self.n_ctn_states,int(self.max_n_ctn[svc])) 
        ret *= compute_formula(self.n_req_states,int(2*self.queue_size))
        return ret


    def _cal_action_mask(self):
        """
        Calculate the action mask for the environment.
        This method fills the action mask with 1s initially and then sets specific
        ranges to 0 based on the service and trans constraints. The action mask
        is used to determine which actions are valid in the current state of the environment.
        The method iterates over the services and transactions to calculate the minimum
        and maximum indices for the action mask that should be set to 0. The calculation
        involves the number of services (`n_svc`), the maximum number of containers per
        service (`max_n_ctn`), and the number of transactions (`n_trans`).
        The action mask is updated in a reversed order of services, and the multiplicative
        factor (`mul`) is adjusted accordingly to ensure the correct indices are calculated.
        
        Theory:
        - https://en.wikipedia.org/wiki/Row-_and_column-major_order
        - https://en.wikipedia.org/wiki/Space-filling_curve 

        Returns:
            None
        """
        self.action_mask.fill(1)
        mul = 1
        for svc in range(self.n_svc-1):
            mul *= (self.max_n_ctn[svc]*self.n_trans + 1)
        
        for svc in reversed(range(self.n_svc)):
            for trans in range(1,self.n_trans+1):
                min_lo_idx = mul*(self._cont_st_mtx[svc][self.TRANS_ST_MAPPING[trans]] + 1 + (trans-1)*self.max_n_ctn[svc])
                max_lo_idx = mul*trans*self.max_n_ctn[svc]
                if min_lo_idx <= max_lo_idx:
                    self.action_mask[min_lo_idx:(max_lo_idx+1)] = 0
            if svc > 0:
                mul //= (self.max_n_ctn[svc-1]*self.n_trans + 1)
            
                 
    def gen_rand_cont_mtx(self):
        ret = np.zeros(shape=(len(self.max_n_ctn), self.n_ctn_states),dtype=np.int64)     
        for svc in range(self.n_svc):
            tmp = self.max_n_ctn[svc]
            for state in range(self.n_ctn_states-2):
                if tmp > 0 :
                    ret[svc][state] = np.random.randint(0,tmp)
                    tmp -= ret[svc][state]
            ret[svc][self.n_ctn_states-2] = tmp
        return ret
    

    def get_res_profile(self, profiling_path, custom_profiling):
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
        self.cal_obs_mtx()   
        return gym.spaces.flatten(self.raw_state_space,self.obs_mtx)


    def _get_reward(self):
        self.step_rwd = self.rwd_add + self.profit - (self.delay_coff*self.delay_pen + self.aban_coff*self.aban_pen + self.en_coff*self.en_cost)
        return self.step_rwd
    
    def reset(self, seed=42, options=None):
        '''
        Initialize the environment
        '''
        super().reset(seed=seed) # We need the following line to seed self.np_random
        
        self.now = 0  # Start at time 0
        self.ptu_res_usage.fill(0)

        self._cont_st_mtx = self._cont_st_mtx_init.copy()
        
        # Observation matrices cache
        self.obs_mtx.fill(0)
        self.obs_mtx[:,0:self.n_ctn_states]=self._cont_st_mtx
        
        # self.action_mask.fill(0)
        self._cal_action_mask()
        
        self.queued_reqs = [[] for _ in range(self.n_svc)] 
        self.active_reqs = [[] for _ in range(self.n_svc)] 
        self.done_reqs = [[] for _ in range(self.n_svc)] 
        self.new_reqs = [[] for _ in range(self.n_svc)] 
        self.rej_reqs = [[] for _ in range(self.n_svc)] 
        
        self.total_accepted_req.fill(0)
        self.total_new_req.fill(0)
        self.cur_n_queued_req.fill(0)
        self.cur_n_active_req.fill(0)
        self.total_done_req.fill(0)
        self.total_rej_req.fill(0)
        self.cum_res_usage.fill(0)
        
        self.cu_req_delay = [[] for _ in range(self.n_svc)]
    
        self.truncated = False
        self.terminated = False
        
        observation = self._get_obs()
        
        return observation
     
           
    def _receive_new_requests(self):
        num_new_req = self.traffic_generator.generate_requests(self.queued_reqs,
                                                              now=self.now)
        self.total_new_req += num_new_req
            
    def _set_truncated(self):
        temp = self._cont_st_mtx + self.cur_act_mtx
        
        # temp_current_usage = np.sum(np.dot(self._cont_st_mtx, Container_Resource_Usage),axis=0)
        # # Instantaneous resource consumption due to state transition
        # if (temp_current_usage[Resource_Type.CPU] > self.limited_resource[Resource_Type.CPU]
        #     or temp_current_usage[Resource_Type.RAM] > self.limited_resource[Resource_Type.RAM]):
        #     # If instantaneous resource consumption exceeds the limit, state transition is not allowed
        #     self.cur_act_mtx.fill(0)
        # else: 
        #     pass

        if (np.any(temp < 0)):
            # If the number of containers is less than 0, state transition is not allowed
            self.truncated = True
            self.truncated_reason = "Wrong number action"
            print("reason: ", self.truncated_reason)
            print("container matrix: ", self._cont_st_mtx)
            print("action matrix: ", self.cur_act_mtx)
            print("action index: ", self.cur_act_idx)
            print("action mask value: ", self.action_mask[self.cur_act_idx])
            print("now: ", self.now)
            # print(self.cur_act_mtx)
            # print(self.cur_act_idx)
            # print(self.action_mask[self.cur_act_idx])
            # print(self.cur_act_idx)
            # print(self.now)
        else: 
            pass
            
              
    def _set_terminated(self):
        if (self.now >= self.cont_span):
            self.terminated = True                      
            
    def _do_env(self):
        """
        Simulates the environment in a step step_interval, handling state transitions, 
        resource usage, and request processing by looping over each time unit in step step_interval.
        The function performs the following tasks in each time unit:
        - Receives new requests at each time step.
        - Calculates the resource usage for each service.
        - Handles state transitions of containers.
        - Processes queued requests, either rejecting them if they have timed out or accepting 
          them if resources are available.
        - Processes active requests, updating their state when they are completed.
        - Calculates the profit from accepted requests and the energy cost based on power usage.
        """
        pos_act_mtx = self.cur_act_mtx * (self.cur_act_mtx > 0)
        neg_act_mtx = self.cur_act_mtx * (self.cur_act_mtx < 0)
        
        self._cont_st_mtx += neg_act_mtx
        relative_time = 0
        while relative_time < self.step_interval:
            self._receive_new_requests()
            # Reset per time unit resource usage matrix
            self.ptu_res_usage  = np.sum(np.dot(self._cont_st_mtx, self.cont_res_usage),axis=0)
            for svc in range(self.n_svc):
                trans_num  = self.fmt_act[0][svc] # Number of containers for each service in action matrix
                trans_type = self.fmt_act[1][svc] # Transition type for each service in action matrix
                
                # Complete state transition
                if relative_time == np.ceil(self.trans_cost[trans_type][Resource_Type.Time]):
                    self._cont_st_mtx[svc] += pos_act_mtx[svc]
                    
                elif relative_time < np.ceil(self.trans_cost[trans_type][Resource_Type.Time]):
                    # Instantaneous resource consumption due to state transition   
                    self.ptu_res_usage[Resource_Type.CPU] += self.trans_cost[trans_type][Resource_Type.CPU]*trans_num 
                    self.ptu_res_usage[Resource_Type.RAM] += self.trans_cost[trans_type][Resource_Type.RAM]*trans_num 
                    self.ptu_res_usage[Resource_Type.Power] += self.trans_cost[trans_type][Resource_Type.Power]*trans_num
                
                # Handle requests in queue
                for req in self.queued_reqs[svc][:]:
                    # Release requests that have timed out
                    if self.now == np.ceil(req.max_queue_delay + req.enq_ts):
                        req.set_state(Request_States.Rejected)
                        req.set_exit_ts(self.now)
                        self.rej_reqs[svc].append(req)
                        self.queued_reqs[svc].remove(req)
                        
                        # Abandon penalty is applied only once at the time the request times out and is rejected by the system
                        self.aban_pen += self.cont_res_usage[Container_States.Active][Resource_Type.RAM]*self.ram_profit*req.active_time
                        self.aban_pen += self.cont_res_usage[Container_States.Active][Resource_Type.CPU]*self.cpu_profit*req.active_time
                    else:
                        # If there are available resources, push the request into the system
                        if self._cont_st_mtx[svc][Container_States.Warm_CPU] > 0:
                            req.set_state(Request_States.In_System)
                            req.set_deq_ts(self.now)
                            self.total_accepted_req [svc] += 1
                            self.active_reqs[svc].append(req)
                            self.queued_reqs[svc].remove(req)
                            self._cont_st_mtx[svc][Container_States.Active] += 1
                            self._cont_st_mtx[svc][Container_States.Warm_CPU] -= 1
                            
                            # Delay penalty is applied only once at the time the request is accepted by the system
                            delay_time = req.deq_ts - req.enq_ts
                            self.delay_pen += self.cont_res_usage[Container_States.Active][Resource_Type.RAM]*self.ram_profit*delay_time
                            self.delay_pen += self.cont_res_usage[Container_States.Active][Resource_Type.CPU]*self.cpu_profit*delay_time
                            
                            self.req_delay[svc].append(delay_time)

                # Handle requests in system
                for req in self.active_reqs[svc][:]:
                    # Resource consumption by request
                    # self.ptu_res_usage  += REQ_RES_USAGE[svc]
                    # Release requests that have been completed
                    if req.active_time == np.ceil(self.now - req.deq_ts):
                        req.set_state(Request_States.Done)
                        req.set_exit_ts(self.now)
                        self.done_reqs[svc].append(req)
                        self.active_reqs[svc].remove(req)
                        self._cont_st_mtx[svc][Container_States.Active] -= 1
                        self._cont_st_mtx[svc][Container_States.Warm_CPU] += 1
                
                # Profit of requests accepted into the system in 1 second
                self.profit += self.cont_res_usage[Container_States.Active][Resource_Type.RAM]*self.ram_profit*self._cont_st_mtx[svc][Container_States.Active]
                self.profit += self.cont_res_usage[Container_States.Active][Resource_Type.CPU]*self.cpu_profit*self._cont_st_mtx[svc][Container_States.Active]
            
            self.en_cost += self.ptu_res_usage [Resource_Type.Power]*self.en_price 
            self.cum_res_usage += self.ptu_res_usage 
                
            self.now += 1
            relative_time += 1

    
    def _cal_sys_eval(self):
        for svc in range(self.n_svc):    
            self.cur_n_queued_req[svc] = len(self.queued_reqs[svc])
            self.cur_n_active_req[svc] = self._cont_st_mtx[svc][Container_States.Active]
            self.total_done_req[svc] = len(self.done_reqs[svc])
            self.total_rej_req[svc] = len(self.rej_reqs[svc])
            
    def  cal_obs_mtx(self):
        self.obs_mtx[:,0:self.n_ctn_states]=self._cont_st_mtx
        for svc in range(self.n_svc):
            self.obs_mtx[svc][self.n_ctn_states+Request_States.In_Queue] = len(self.queued_reqs[svc])
  
    
    def act_idx_to_mtx(self,act_idx):
        """
        Convert an action index to an action matrix.
        This function takes an action index and converts it into a matrix representation
        of the action. The matrix has two rows: the first row represents the number of containers
        for each service, and the second row represents the transition type for each service.
        Parameters:
        act_idx (int): The action index to be converted.
        Returns:
        np.ndarray: A 2xN matrix where N is the number of services. The first row contains the
                    number of containers for each service, and the second row contains the
                    transition type for each service.
        """
        act_mtx = np.zeros((2,self.n_svc),dtype=np.int32)
        lo_idx = 0
        mul = 1 
        for svc in range(self.n_svc-1):
            mul *= (self.max_n_ctn[svc]*self.n_trans + 1)
            
        for svc in reversed(range(self.n_svc)):
            lo_idx = act_idx // mul # local index of an action in action subspace of a service
            if lo_idx == 0:
                print("action 0 ne")
                act_mtx[0][svc] = 0
                act_mtx[1][svc] = 0
            else:
                act_mtx[0][svc] = ((lo_idx-1) % self.max_n_ctn[svc]) + 1
                act_mtx[1][svc] = ((lo_idx-1) // self.max_n_ctn[svc]) + 1
            act_idx %= mul
            mul //= (self.max_n_ctn[svc-1]*self.n_trans + 1)
            
        act_mtx = act_mtx.reshape(2,self.n_svc)
        act_coff = np.diag(act_mtx[0])
        act_unit = []
        for svc in act_mtx[1]:
            act_unit.append(self.TRANS[svc])
        
        self.cur_act_mtx = act_coff @ act_unit
        self.cur_act_idx = act_idx
        return act_mtx
        
    def _clear_cache(self):
        self.new_reqs = [[] for _ in range(self.n_svc)] 
        self.done_reqs = [[] for _ in range(self.n_svc)] 
        self.rej_reqs = [[] for _ in range(self.n_svc)] 
        self.req_delay = [[] for _ in range(self.n_svc)] 
        self.obs_mtx.fill(0)
        self.action_mask.fill(0)
        self.step_rwd = 0
        self.aban_pen = 0
        self.delay_pen = 0
        self.profit = 0
        self.en_cost = 0
        self.total_new_req.fill(0)
        self.cur_n_queued_req.fill(0)
        self.cur_n_active_req.fill(0)
        self.total_done_req.fill(0)
        self.total_accepted_req.fill(0)
        self.total_rej_req.fill(0)
        self.truncated = False
        self.terminated = False
                 
    def _pre_step(self,action):
        self._clear_cache()
        self.fmt_act = self.act_idx_to_mtx(action)
        self._set_terminated()
        self._set_truncated()
        
        
    def step(self, action):
        self._pre_step(action)
        self._do_env()   
        self._cal_sys_eval()
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
            "step_interval": self.now // self.step_interval,
            "action_number": self.cur_act_idx,
            "action_matrix": self.cur_act_mtx.tolist(),  
            "containers_state_after_action": self._cont_st_mtx.tolist(),
            "cumulative_number_accepted_request": self.total_accepted_req.tolist(),
            "new_requests": self.total_new_req.tolist(),
            "queue_requests": self.cur_n_queued_req.tolist(),
            "system_requests": self.cur_n_active_req.tolist(),
            "done_requests": self.total_done_req.tolist(),
            "timeout_requests": self.total_rej_req.tolist(),
            "request_delay": self.req_delay,
            "step_reward": self.step_rwd,
            "profit": self.profit,
            "abandone_penalty value": self.aban_coff*self.aban_pen,
            "delay_penalty value": self.delay_coff*self.delay_pen,
            "energy_cost value": self.en_coff*self.en_cost,
            "energy_consumption": self.cum_res_usage[Resource_Type.Power],
            "ram_consumption": self.cum_res_usage[Resource_Type.RAM],
            "cpu_consumption": self.cum_res_usage[Resource_Type.CPU],
            "per_second_energy_usage": self.ptu_res_usage [Resource_Type.Power],
            "per_second_ram_usage": self.ptu_res_usage [Resource_Type.RAM],
            "per_second_cpu_usage": self.ptu_res_usage [Resource_Type.CPU]
        } 
        return log_data
    

