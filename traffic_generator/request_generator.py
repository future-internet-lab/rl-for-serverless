import numpy as np
from abc import ABC, abstractmethod
from rlss_env.request import Request
import rlss_env.profiling as profiling
from scipy.optimize import minimize
import pandas as pd
from scipy import stats

def truncated_lognormal_single_sample_fast(mu, sigma, min_val, max_val):
    """
    Generate a single sample from a truncated log-normal distribution within a specified range 
    using inverse transform sampling.
    
    Parameters:
        mu (float): The mean of the underlying normal distribution.
        sigma (float): The standard deviation of the underlying normal distribution.
        min_val (float): The minimum value for truncation.
        max_val (float): The maximum value for truncation.
    
    Returns:
        sample: A single sample from the truncated log-normal distribution.
    """
    # Calculate the CDF values for min_val and max_val in the log-normal distribution
    cdf_min = stats.lognorm.cdf(min_val, sigma, scale=np.exp(mu))
    cdf_max = stats.lognorm.cdf(max_val, sigma, scale=np.exp(mu))
    
    # Generate a uniform random value in the range [cdf_min, cdf_max]
    u = np.random.uniform(cdf_min, cdf_max)
    
    # Use the inverse CDF (ppf) to generate the corresponding sample
    sample = stats.lognorm.ppf(u, sigma, scale=np.exp(mu))
    
    return sample

def estimate_dist(avg, percentiles):
    """
    Estimate the parameters of a log-normal distribution using the method of moments.
    
    Parameters:
        avg (float): The average value.
        percentiles (dict): A dictionary containing the percentiles ('0', '25', '50', '75', '99', '100').
    
    Returns:
        mu (float): The mean of the underlying normal distribution.
        sigma (float): The standard deviation of the underlying normal distribution.
    """
    # Calculate mean and variance from the average and percentiles
    mean = avg
    # Estimate variance using interquartile range (IQR) between 25th and 75th percentiles
    iqr = percentiles['75'] - percentiles['25']
    variance = (iqr / 1.349) ** 2  # Approximate variance from IQR
    
    # Calculate the parameters mu and sigma using the method of moments
    mu = np.log(mean**2 / np.sqrt(variance + mean**2))
    sigma = np.sqrt(np.log(variance / mean**2 + 1))
    return mu, sigma

class TrafficGenerator(ABC):
    def create_request(self, request_type, now, active_time):
        return Request(type=request_type, enq_ts=int(now), max_queue_delay=self.max_queue_delay[request_type], active_time=active_time)

    @abstractmethod
    def determine_active_time(self, request_type):
        pass
    
    @abstractmethod
    def generate_requests(self, queue, now):
        pass

class PoissonGenerator(TrafficGenerator):
    def __init__(self, size=1, avg_requests_per_second=1, max_queue_delay=[20], max_rq_active_time={"type": "random", "value": [60]}):
        self.name = "Poisson"
        self.num_services = size
        self.avg_requests_per_second = avg_requests_per_second
        self.max_queue_delay = max_queue_delay
        self.max_rq_active_time = max_rq_active_time
        # super().__init__(size, avg_requests_per_second, max_queue_delay, max_rq_active_time)

    def ran_norm_gen(self, mean, std_dev):
        value = np.random.normal(loc=mean, scale=std_dev)
        int_value = round(value)
        positive_int_value = max(1, int_value)
        return positive_int_value
    
    def determine_active_time(self, request_type):
        if self.max_rq_active_time["type"] == "random":
            return self.ran_norm_gen(self.max_rq_active_time["value"][request_type], self.max_rq_active_time["value"][request_type]/2)
        else:
            return self.max_rq_active_time["value"][request_type] or profiling.REQ_ACTIVE_TIME[request_type]
        
    def generate_requests(self, queue, now):
        rng = np.random.default_rng()
        new_rqs = np.zeros(self.size, dtype=np.uint32)    

        for request_type in range(self.num_services):
            num_new_rqs = rng.poisson(self.avg_requests_per_second)  
            for _ in range(num_new_rqs):
                active_time = self.determine_active_time(request_type) 
                rq = self.create_request(request_type, now, active_time)  
                queue[request_type].append(rq)  
                new_rqs[request_type] += 1   
        
        return new_rqs

class RealTraceGenerator(TrafficGenerator):
    def __init__(self, active_time_stats_file, arrival_request_stats_file, max_queue_delay=[4], num_services=1):
        self.num_services = num_services
        self.max_queue_delay= max_queue_delay
        self.num_arrival_stats = []
        self.mu = []
        self.sigma = []
        self.minimum = []
        self.maximum = []
        self._get_active_time_stats(active_time_stats_file)
        self._get_num_arrival_stats(arrival_request_stats_file)
        # super().__init__(size, avg_requests_per_second, max_queue_delay, max_rq_active_time)
    
    def _get_active_time_stats(self, active_time_stats_file):
        if active_time_stats_file:
            data = pd.read_csv(active_time_stats_file)
            for i in range(self.num_services):
                
                percentiles = {
                    '0': data.iloc[i]['percentile_Average_0'],
                    '1': data.iloc[i]['percentile_Average_1'],
                    '25': data.iloc[i]['percentile_Average_25'],
                    '50': data.iloc[i]['percentile_Average_50'],
                    '75': data.iloc[i]['percentile_Average_75'],
                    '99': data.iloc[i]['percentile_Average_99'],
                    '100': data.iloc[i]['percentile_Average_100']
                }
                average = data.iloc[i]['Average']
                mu, sigma = estimate_dist(average, percentiles)
                self.mu.append(mu)
                self.sigma.append(sigma)
                self.minimum.append(data.iloc[i]['Minimum'])
                self.maximum.append(data.iloc[i]['Maximum'])
                
    def _get_num_arrival_stats(self, arrival_request_per_minute_file):
        if arrival_request_per_minute_file:
            data = pd.read_csv(arrival_request_per_minute_file)
            for i in range(self.num_services):
                self.num_arrival_stats.append(data.iloc[i].dropna().tolist())
                self.num_arrival_stats[i] = [x / 60 for x in self.num_arrival_stats[i]]
    
    def determine_active_time(self, service_index):
        return truncated_lognormal_single_sample_fast(self.mu[service_index], 
                                                      self.sigma[service_index], 
                                                      self.minimum[service_index], 
                                                      self.maximum[service_index])
    
    def generate_requests(self, queue, now):
        rng = np.random.default_rng()
        current_minute = int(np.floor(now / 60))
        new_rqs = np.zeros(self.num_services, dtype=np.uint32)    

        for request_type in range(self.num_services):
            num_requests = rng.poisson(self.num_arrival_stats[request_type][current_minute])
            for _ in range(num_requests):
                active_time = self.determine_active_time(request_type) 
                request = self.create_request(request_type, now, active_time)  
                queue[request_type].append(request)  
                new_rqs[request_type] += 1   
        return new_rqs
