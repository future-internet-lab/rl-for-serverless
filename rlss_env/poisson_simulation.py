import numpy as np
import matplotlib.pyplot as plt

class Poisson():
    def __init__(self, rate, time_duration):
        super(Poisson, self).__init__()
        self.rate = rate
        self.time_duration = time_duration
        
    def generate_poisson_events(self):
        # Tính tổng số sự kiện bằng phân phối Poisson
        num_events = np.random.poisson(self.rate * self.time_duration)
        
        # Tạo ra thời gian đến giữa các sự kiện bằng phân phối mũ với giá trị trung bình là 1.0 / rate
        inter_arrival_times = np.random.exponential(1.0 / self.rate, num_events)
        
        # Cộng dồn thời gian giữa các lần đến để có được thời gian đến của sự kiện
        event_times = np.cumsum(inter_arrival_times)
        
        # Trả về số lượng sự kiện, thời gian sự kiện và thời gian giữa các lần đến tương ứng
        print(num_events, event_times, inter_arrival_times)
        return num_events, event_times, inter_arrival_times
    
    def plot_non_sequential_poisson(self, num_events, event_times, inter_arrival_times):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Poisson Process Simulation (λ = {self.rate}, Duration = {self.time_duration} seconds)\n', fontsize=16)

        axs[0].step(event_times, np.arange(1, num_events + 1), where='post', color='blue')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Event Number')
        axs[0].set_title(f'Poisson Process Event Times\nTotal: {num_events} events\n')
        axs[0].grid(True)

        axs[1].hist(inter_arrival_times, bins=20, color='green', alpha=0.5)
        axs[1].set_xlabel('Inter-Arrival Time')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title(f'Histogram of Inter-Arrival Times\nMEAN: {np.mean(inter_arrival_times):.2f} | STD: {np.std(inter_arrival_times):.2f}\n')
        axs[1].grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
    def plot_sequential_poisson(self, num_events_list, event_times_list, inter_arrival_times_list):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Poisson Process Simulation (Duration = {self.time_duration} seconds)\n', fontsize=16)

        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Event Number')
        axs[0].set_title(f'Poisson Process Event Times')
        axs[0].grid(True)

        axs[1].set_xlabel('Inter-Arrival Time')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title(f'Histogram of Inter-Arrival Times')
        axs[1].grid(True, alpha=0.5)

        color_palette = plt.get_cmap('tab20')
        colors = [color_palette(i) for i in range(len(self.rate))]

        for n, individual_rate in enumerate(self.rate):
            num_events = num_events_list[n]
            event_times = event_times_list[n]
            inter_arrival_times = inter_arrival_times_list[n]

            axs[0].step(event_times, np.arange(1, num_events + 1), where='post', color=colors[n], label=f'λ = {individual_rate}, Total Events: {num_events}')
            axs[1].hist(inter_arrival_times, bins=20, color=colors[n], alpha=0.5, label=f'λ = {individual_rate}, MEAN: {np.mean(inter_arrival_times):.2f}, STD: {np.std(inter_arrival_times):.2f}')

        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def poisson_simulation(self, show_visualization=True):
        if isinstance(self.rate, int):
            num_events, event_times, inter_arrival_times = self.generate_poisson_events()
            
            if show_visualization:
                self.plot_non_sequential_poisson(num_events, event_times, inter_arrival_times)
            else:
                return num_events, event_times, inter_arrival_times

        elif isinstance(self.rate, list):
            num_events_list = []
            event_times_list = []
            inter_arrival_times_list = []

            for individual_rate in self.rate:
                num_events, event_times, inter_arrival_times =self.generate_poisson_events()
                num_events_list.append(num_events)
                event_times_list.append(event_times)
                inter_arrival_times_list.append(inter_arrival_times)

            if show_visualization:
                self.plot_sequential_poisson(num_events_list, event_times_list, inter_arrival_times_list)
            else:
                return num_events_list, event_times_list, inter_arrival_times_list
            
# Example usage
if __name__ == "__main__":
    psim_example = Poisson(rate=640/3600, time_duration=86400)
    num_events, event_times, inter_arrival_times = psim_example.generate_poisson_events()
    print(num_events)
    print(event_times[1])
    print(event_times[0] + inter_arrival_times[1])
    # psim_example.poisson_simulation()