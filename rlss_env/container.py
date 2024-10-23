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
    def __init__(self, type: int):
        self.type = type
        self.state = 0 
        self.ram_usage = 0
        self.cpu_usage = 0
        self.power_usage = 0
        self.is_handling_request = False 
        self.set_state()

    def set_state(self, state = 0, ram_usage = 0, cpu_usage = 0, power_usage = 0):
        self.state = state
        self.ram_usage = ram_usage
        self.cpu_usage = cpu_usage
        self.power_usage = power_usage
        
    def set_is_handling_request(self, a: bool):
        self.is_handling_request = a
    