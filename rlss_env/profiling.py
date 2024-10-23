import numpy as np

# Helper function to generate random times using a uniform distribution
def uniform_random_time(a, b, step=1):
    possible_values = np.arange(a, b + step, step)
    return np.random.choice(possible_values)

class Resource_Type:
    RAM = 0
    CPU = 1
    Power = 2
    Time = 3


# [RAM, CPU, Power] 
REQ_RES_USAGE = np.array([np.array([10, 10, 100]),
                          np.array([20, 20, 200]),
                          np.array([30, 30, 300]),
                          np.array([40, 40, 400])])

REQ_ACTIVE_TIME = np.array([240, 360, 480, 600])

# Generate transition cost for moving to other states
# [RAM, CPU, Power, Time]  
def generate_trans_cost():
    def transition_no_change():
        return np.array([0, 0, 0, 0])

    def transition_N_to_L0():
        time = uniform_random_time(3, 8)
        return np.array([0, 0, 5 * time, time])

    def transition_N_to_L2():
        time_1 = uniform_random_time(3, 8)
        time_2 = uniform_random_time(5, 60)
        time_3 = uniform_random_time(3, 5, step=0.5)
        total_power = 5 * time_1 + 40 * (time_2 + time_3)
        total_time = time_1 + time_2 + time_3
        return np.array([0, 0, total_power, total_time])

    def transition_L0_to_N():
        time = uniform_random_time(1, 5)
        return np.array([0, 0, 5 * time, time])

    def transition_L0_to_L1():
        time = uniform_random_time(5, 60)
        return np.array([0, 0, 40 * time, time])

    def transition_L0_to_L2():
        time_1 = uniform_random_time(5, 60)
        time_2 = uniform_random_time(3, 5, step=0.5)
        total_power = 40 * (time_1 + time_2)
        total_time = time_1 + time_2
        return np.array([0, 0, total_power, total_time])

    def transition_L1_to_L0():
        time = uniform_random_time(1, 5)
        return np.array([0, 0, 5 * time, time])

    def transition_L1_to_L2():
        time = uniform_random_time(3, 5, step=0.5)
        return np.array([0, 0, 40 * time, time])

    def transition_L2_to_L1():
        time = uniform_random_time(30, 40)
        return np.array([0, 0, 10 * time, time])

    transitions_cost = np.array([
        transition_no_change(),          # No change
        transition_N_to_L0(),            # N -> L0
        transition_N_to_L2(),            # N -> L2
        transition_L0_to_N(),            # L0 -> N
        transition_L0_to_L1(),           # L0 -> L1
        transition_L0_to_L2(),           # L0 -> L2
        transition_L1_to_L0(),           # L1 -> L0
        transition_L1_to_L2(),           # L1 -> L2
        transition_L2_to_L1()            # L2 -> L1
    ])

    return transitions_cost


# Generate resource usage for staying in each state
# [RAM, CPU, Power]
def generate_container_resource_usage():
    def state_N():
        return np.array([0, 0, 0])

    def state_L0():
        return np.array([0, 0, 0])

    def state_L1():
        return np.array([0, 0, 0])

    def state_L2():
        time = uniform_random_time(5, 60)
        cpu_percent = 0.05
        power = cpu_percent * 200
        return np.array([20 * time, cpu_percent, power])

    def state_A():
        time = uniform_random_time(5, 60)
        cpu_percent = 0.05 + (0.1 * time)
        power = cpu_percent * 200
        return np.array([20 * time, cpu_percent, power])

    container_resource_usage = np.array([
        state_N(),   # N
        state_L0(),  # L0
        state_L1(),  # L1
        state_L2(),  # L2
        state_A()    # A
    ])

    return container_resource_usage
