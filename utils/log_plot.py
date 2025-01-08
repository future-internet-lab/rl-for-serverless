from rlss_env.container import Container_States as CS
import os 
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle


def append_to_json(log_data, log_file):
    if os.path.exists(log_file):
        with open(log_file, 'a') as json_file:
            json_file.write(json.dumps(log_data, indent=4, separators=(',', ': ')))
    else:
        with open(log_file, 'w') as json_file:
            json_file.write(json.dumps(log_data, indent=4, separators=(',', ': ')))

def append_to_pickle(log_data, log_file):
    if os.path.exists(log_file):
        with open(log_file, 'ab') as pkl_file:
            pickle.dump(log_data, pkl_file)
    else:
        with open(log_file, 'wb') as pkl_file:
            pickle.dump(log_data, pkl_file)


def calc_accept_ratio(
    in_sys_rqs: list, 
    done_rqs: list, 
    in_queue_rqs: list, 
    new_rqs: list
) -> list:
    acceptance_ratio = []
    
    for i in range(len(new_rqs)):
        in_sys = in_sys_rqs[i]
        done = done_rqs[i]
        prev_in_sys = in_sys_rqs[i - 1] if i > 0 else 0
        prev_in_queue = in_queue_rqs[i - 1] if i > 0 else 0
        new_requests = new_rqs[i]
        denominator = prev_in_queue + new_requests
        if denominator == 0:
            ratio = 0
        else:
            ratio = (in_sys + done - prev_in_sys) / denominator
        acceptance_ratio.append(ratio)
    
    return acceptance_ratio


def plot_log_fig(log_file, training_num, step_interval_value, num_service):
    log_folder = os.path.dirname(log_file)
    with open(log_file, 'r') as json_file:
        data = json.load(json_file)

    e = 0
    for key, ep in data.items():
        new_rqs =  [[] for _ in range(num_service)]
        in_queue_rqs = [[] for _ in range(num_service)]
        in_sys_rqs = [[] for _ in range(num_service)]
        done_rqs = [[] for _ in range(num_service)]
        rq_delays = [[] for _ in range(num_service)]
        container_states = [[] for _ in range(num_service)]

        rewards = []
        energy_consumptions = []
        s = 0
        for step in ep:
            rewards.append(step["step_reward"])
            energy_consumptions.append(step["energy_consumption"])
            for service in range(num_service):
                new_rqs[service].append(step["new_requests"][service])
                in_queue_rqs[service].append(step["queue_requests"][service])
                in_sys_rqs[service].append(step["system_requests"][service])
                done_rqs[service].append(step["done_requests"][service])
                rq_delays[service].extend(step["request_delay"][service])
                container_states[service].append(step["containers_state_after_action"][service])
            s += 1
                
        
        step_intervals = np.arange(s) * step_interval_value
        # Plot acceptance ratio
        plt.figure()
        for i in range(num_service):
            acpt_ratios =calc_accept_ratio(in_sys_rqs[service], done_rqs[service], in_queue_rqs[service], new_rqs[service])
            plt.plot(step_intervals, acpt_ratios, label=f'Service {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceptance Ratio')
        plt.title('Avg Acceptance Ratio')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(log_folder, f'acpt_ratio_episode_{e}.png'))  
        plt.close()
        
        # Plot bar request delay
        plt.figure()
        for i in range(num_service): 
            if rq_delays[i]:
                min_value = min(rq_delays[i])
                max_value = max(rq_delays[i])
                mean_value = np.mean(rq_delays[i])

                values = [mean_value, min_value, max_value]
                labels = [f'Mean service {e}', f'Min service {e}', f'Max service{e}']

                bars = plt.bar(labels, values)

                for bar in bars:
                    yval = bar.get_height()  
                    plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", ha='center', va='bottom') 
        plt.title(f'Delay time of accepted request')
        plt.ylabel('Delay time (s)')
        plt.savefig(os.path.join(log_folder, f'bar_delay_episode_{e}.png'))  
        plt.close()
        
        # Plot boxplot request delay
        plt.figure()
        bars = plt.boxplot(rq_delays)

        plt.title(f'Delay time of accepted request')
        plt.ylabel('Delay time (s)')
        plt.xlabel('Service')
        plt.savefig(os.path.join(log_folder, f'boxplot_delay_episode_{e}.png')) 
        plt.close()
        
        # # Plot line request delay
        # for i in range(num_service):
        #     plt.figure()
        #     plt.plot(rq_delays[i], label=f'Service {i+1}')

        # plt.title(f'Delay time accepted request')
        # plt.ylabel('Delay time (s)')
        # plt.xlabel('')
        # plt.savefig(os.path.join(log_folder, f'line_delay_{e}.png')) 

        # Plot reward
        plt.figure()
        plt.plot(step_intervals, rewards, label='Avg Rewards', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Rewards')
        plt.title('Reward over step')
        plt.grid(True)
        plt.savefig(os.path.join(log_folder, f'reward_episode_{e}.png'))  
        plt.close()

        # Plot Energy consumption
        plt.figure()
        plt.plot(step_intervals, energy_consumptions, label='Energy Consumption', color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Consumption (J)')
        plt.title('Cummulative Energy Consumption')
        plt.grid(True)
        plt.savefig(os.path.join(log_folder, f'energy_cons_episode_{e}.png'))  
        plt.close()
        
        num_ctn_states = len(CS.State_Name)
        
        # Plot area container state
        for i in range(num_service):
            plt.figure()
            plt.stackplot(step_intervals, np.array(container_states[i]).T, labels=[f'{CS.State_Name[i]}' for i in range(num_ctn_states)])
            plt.xlabel('Time (s)')
            plt.ylabel('Number container')
            plt.title('Ratio between container states of service {}'.format(service))
            plt.legend()
            plt.savefig(os.path.join(log_folder, 'cont_state_service_{}_episode_{}.png'.format(service,e))) 
            plt.close()
        
        # Plot areline container state
        for i in range(num_service):
            plt.figure()
            for k in range(num_ctn_states):
                plt.plot(step_intervals, np.array(container_states[i]).T[k], label=f'{CS.State_Name[k]}')
            plt.xlabel('Time (s)')
            plt.ylabel('Number container')
            plt.title('Ratio between container states of service {}'.format(service))
            plt.legend()
            plt.savefig(os.path.join(log_folder, 'line_cont_state_service_{}_episode_{}.png'.format(service,e))) 
            plt.close()
        
        e += 1

