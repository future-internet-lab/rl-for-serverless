import re
from rlss_envs import Container_States
import os
import matplotlib.pyplot as plt
import numpy as np

num_ctn_states = len(Container_States.State_Name)

def plot_log_fig(log_folder):
    log_file = os.path.join(log_folder, 'log.txt')

    with open(log_file, 'r') as file:
        log_data = file.read()
    
    training_num_pattern = re.compile(r'Test trainned model (\d+) times') 
    training_num_match = training_num_pattern.search(log_data)   
    training_num = int(training_num_match.group(1))
    
    service_num_pattern = re.compile(r'"num_service": (\d+),')
    service_num_match = service_num_pattern.search(log_data)   
    service_num = int(service_num_match.group(1))

    timestep_pattern = re.compile(r'"timestep": (\d+),')
    timestep_match = timestep_pattern.search(log_data)
    if timestep_match:
        timestep_value = int(timestep_match.group(1))

    timestep_blocks = re.findall(r'SYSTEM EVALUATION PARAMETERS IN TIMESTEP \d+:.*?(?=SYSTEM EVALUATION PARAMETERS IN TIMESTEP \d+:|\Z)', log_data, re.S)

    new_rqs = []
    in_queue_rqs = []
    in_sys_rqs = []
    done_rqs = []
    rewards = []
    energy_consumptions = []
    cu_rq_delays = []
    container_states = []
    cu_accepted_rqs = []

    for i, block in enumerate(timestep_blocks, start=1):
        container_state_match = re.search(r"Containers state after action:\s*\[\s*((?:\[\s*[\d\s]+]\s*)+)\]", block)
        container_state = [list(map(int, re.findall(r'\d+', row))) for row in container_state_match.group(1).split(']\n')]
        
        cu_accepted_rq_match = re.search(r"Cumulative number accpeted request\s*:\s*\[\s*([\d\s]+)\]", block)
        cu_accepted_rq = np.array(list(map(int, cu_accepted_rq_match.group(1).split()))) if cu_accepted_rq_match else None
        
        new_rq_match = re.search(r'Number new request\s*:\s*\[\s*([\d\s]+)\]', block)
        new_rq = np.array(list(map(int, new_rq_match.group(1).split()))) if new_rq_match else None
        
        in_queue_rq_match = re.search(r'Number in queue request\s*:\s*\[\s*([\d\s]+)\]', block)
        in_queue_rq = np.array(list(map(int, in_queue_rq_match.group(1).split()))) if in_queue_rq_match else None
        
        in_sys_rq_match = re.search(r'Number in system request\s*:\s*\[\s*([\d\s]+)\]', block)
        in_sys_rq = np.array(list(map(int, in_sys_rq_match.group(1).split()))) if in_sys_rq_match else None
        
        done_rq_match = re.search(r'Number done system request\s*:\s*\[\s*([\d\s]+)\]', block)
        done_rq = np.array(list(map(int, done_rq_match.group(1).split()))) if done_rq_match else None
        
        cu_rq_delay_match = re.search(r"Cumulative request delay\s*:\s*\[\s*([\d\s]+)\]", block)
        cu_rq_delay = np.array(list(map(int, cu_rq_delay_match.group(1).split()))) if cu_rq_delay_match else None
        
        rewards_match = re.search(r'Rewards\s*:\s*([-\d.]+)', block)
        reward = float(rewards_match.group(1)) if rewards_match else None
        
        energy_match = re.search(r"Energy consumption over timestep\s*:\s*([\d\.]+)J", block)
        energy_consumption = float(energy_match.group(1)) if energy_match else None
        
        new_rqs.append(new_rq)
        in_queue_rqs.append(in_queue_rq)
        in_sys_rqs.append(in_sys_rq)
        done_rqs.append(done_rq)
        rewards.append(reward)
        energy_consumptions.append(energy_consumption)
        cu_rq_delays.append(cu_rq_delay)
        container_states.append(container_state)
        cu_accepted_rqs.append(cu_accepted_rq)

    acceptance_ratio = [
        (in_sys_rqs[i] + done_rqs[i] - (in_sys_rqs[i-1] if i > 0 else 0)) /
        ((in_queue_rqs[i-1] if i > 0 else 0) + new_rqs[i])
        for i in range(len(new_rqs))
    ]
    
    energy_consumptions = np.array(energy_consumptions)
    acceptance_ratio = np.array(acceptance_ratio)
    rewards = np.array(rewards)
    cu_rq_delays = np.array(cu_rq_delays)
    container_states = np.array(container_states)
    cu_accepted_rqs = np.array(cu_accepted_rqs)
    num_step = len(energy_consumptions) // training_num
    
    avg_energy_consumptions = [
        np.mean(energy_consumptions[i::num_step]) 
        for i in range(num_step)
    ]

    avg_acceptance_ratio = [
        np.mean(acceptance_ratio[i::num_step],axis=0) 
        for i in range(num_step)
    ]
    avg_acceptance_ratio = np.array(avg_acceptance_ratio)
    
    cu_rq_delays = np.array([np.mean(cu_rq_delays[::num_step],axis=0)])
    cu_accepted_rqs = np.array([np.mean(cu_accepted_rqs[::num_step],axis=0)])
    avg_cu_rq_delay = cu_rq_delays / cu_accepted_rqs
    
    avg_container_state = np.array([
        np.mean(container_states[i::num_step],axis=0) 
        for i in range(num_step)
    ])
    avg_container_state = np.split(avg_container_state,service_num,axis=1)
    avg_container_state = [arr.squeeze(axis=1) for arr in avg_container_state]
  
    avg_rewards = [
        np.mean(rewards[i::num_step]) 
        for i in range(num_step)
    ]
    
    timesteps = np.arange(len(avg_energy_consumptions)) * timestep_value
    
    # Plot acceptance ratio
    plt.figure()
    for i in range(service_num):
        plt.plot(avg_acceptance_ratio[:, i], label=f'Service {i+1}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceptance Ratio')
    plt.title('Avg Acceptance Ratio Over {} Episodes'.format(training_num))
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_folder, 'acceptance_ratio.png'))  
    
    # Plot avg request delay 
    plt.figure()
    plt.bar(np.arange(1,service_num+1),avg_cu_rq_delay[0, :])
    plt.xlabel('Service')
    plt.ylabel('Delay time per accepted request')
    plt.title('Avg Request Delay Time Over {} Episodes'.format(training_num))
    plt.xticks(np.arange(1,service_num+1))
    plt.savefig(os.path.join(log_folder, 'delay.png'))  

    # Plot reward
    plt.figure()
    plt.plot(timesteps, avg_rewards, label='Avg Rewards Over {} Episodes'.format(training_num), color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rewards')
    plt.title('Avg Rewards Over {} Episodes'.format(training_num))
    plt.grid(True)
    plt.savefig(os.path.join(log_folder, 'rewards_plot.png'))  

    # Plot Energy consumption
    plt.figure()
    plt.plot(timesteps, avg_energy_consumptions, label='Avg Energy Consumption Over {} Episodes'.format(training_num), color='orange')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy Consumption (J)')
    plt.title('Avg Energy Consumption Over {} Episodes'.format(training_num))
    plt.grid(True)
    plt.savefig(os.path.join(log_folder, 'energy_consumption_plot.png'))  
    
    # Plot container state
    for service in range(service_num):
        plt.figure()

        # Plot stacked area chart
        plt.stackplot(timesteps, avg_container_state[service].T, labels=[f'{Container_States.State_Name[i]}' for i in range(num_ctn_states)])

        # Thêm nhãn, tiêu đề, và legend
        plt.xlabel('Time')
        plt.ylabel('Number container')
        plt.title('Ratio between container states of service {}'.format(service))
        plt.legend()
        plt.savefig(os.path.join(log_folder, 'state_service_{}.png'.format(service))) 