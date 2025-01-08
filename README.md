# Serverless optimization using deep reinforcement learning
## Code structure
- `rlss_env/` folder: Implementation of serverless environment base-on Gymnasium library
- `utils/` folder which includes:
    - `models.py`, `dqn_agent.py`: Neural network and Deep Q-learning implementation. Most of source code in these two files comes from https://github.com/MOSAIC-LAB-AALTO/drl_based_trigger_selection
    - Others function to log, debug and support for main function
- `main_dqn.py`: Include main function for training and testing DRL agorithms with serverless environment.
## How to run code?
So far we have only deployed the DQN algorithm for the purpose of testing the environment's behavior. 
To train model, run command:
```
python main_dqn.py -m dqn -p hyperparameters.json
```
After run following command, program will create 
```
your_main_folder/
├── {other files}
├── main_dqn.py
├── result/
    ├── result_{id}/
        ├── dqn_network_
        ├── target_dqn_network_
        ├── hyperparameters.json
        ├── profiling.npz
        └── train/
            ├── live_training.png
            └── log.pkl
        └── test/
```
Trained model's weight is saved in `dqn_network_` and `target_dqn_network_`. System's information and reward of each episode in training process are saved in `log.pkl`. Random resource profiling array is saved in `profiling.npz`. Env and model parameters is saved in `hyperparameters.json`.

You can also open `live_training.png` by vscode to monitor the training process. When the training process is complete, the program will also automatically test model using the saved weights. Testing results is saved in `result/test/` folder.

You can also use saved model's weights of other training times by running following command (replace `n` with number episode you want to test and  `your_main_folder/result/result_{id}` with folder which contains model's weights file).
```
python main_dqn.py -m dqn -p "hyperparameter_json_file" -o n -f "your_main_folder/result/result_{id}"
```

## Configuration file options
The `hyperparameters.json` file contains various settings for training the model. Below are the key options:

- `learning_rate`: The learning rate for the optimizer.
- `gamma`: The discount factor for future rewards.
- `epsilon_start`: The starting value of epsilon for the epsilon-greedy policy.
- `epsilon_end`: The minimum value of epsilon after decay.
- `epsilon_decay`: The rate at which epsilon decays.
- `batch_size`: The number of samples per batch for training.
- `target_update`: The frequency (in episodes) to update the target network.
- `memory_size`: The size of the replay memory.
- `num_episodes`: The number of episodes to train the model.
- `max_steps_per_episode`: The maximum number of steps per episode.

Example `hyperparameters.json`:
```json
{
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "target_update": 10,
    "memory_size": 10000,
    "num_episodes": 500,
    "max_steps_per_episode": 200
}
```

## Environment configuration options
The environment configuration file contains various settings for the serverless environment. Below are the key options:

- `render_mode`: The mode for rendering the environment.
- `num_service`: The number of services.
- `step_interval`: The interval between steps. Time unit depends on how often agent will make an action.
- `num_container`: The number of containers.
- `cont_span`: The span of container activity .
- `traffic_generator`: The type of traffic generator. 'real' or 'simulated'.
- `active_duration_stats_file`: The file path for active time statistics.
- `arrival_request_stats_file`: The file path for arrival request statistics.
- `rq_timeout`: The request timeout.
- `average_requests`: The average number of requests.
- `max_rq_active_duration`: The maximum request active time.
- `energy_price`: The price of energy.
- `ram_profit`: The profit from RAM usage.
- `cpu_profit`: The profit from CPU usage.
- `delay_coff`: The coefficient for delay.
- `aban_coff`: The coefficient for abandonment.
- `energy_coff`: The coefficient for energy.
- `reward_add`: The additional to make reward positive.

Example environment configuration:
```json
{
    "render_mode": null,
    "num_service": 1,
    "step_interval": 120,
    "num_container": [5000],
    "cont_span": 28800,
    "traffic_generator": "real",
    "active_duration_stats_file": "/home/mec/hai/RL-for-serverless/envs/percentile.csv",
    "arrival_request_stats_file": "/home/mec/hai/RL-for-serverless/envs/request.csv",
    "rq_timeout": [2],
    "average_requests": 0.2,
    "max_rq_active_duration": {
        "type": "random",
        "value": [60]
    },
    "energy_price": 8e-06,
    "ram_profit": 1e-05,
    "cpu_profit": 1e-05,
    "delay_coff": 1,
    "aban_coff": 1,
    "energy_coff": 2,
    "reward_add": 0
}
```

## Training configuration options
The training configuration file contains various settings for training the model. Below are the key options:

- `episodes`: The number of episodes to train the model.
- `batch_size`: The number of samples per batch for training.
- `max_env_steps`: The maximum number of steps per episode.
- `batch_update`: The frequency (in steps) to update the batch.
- `replay_buffer_size`: The size of the replay buffer.
- `hidden_size`: The size of the hidden layers.
- `gamma`: The discount factor for future rewards.
- `epsilon`: The starting value of epsilon for the epsilon-greedy policy.
- `eps_decay`: The rate at which epsilon decays.
- `eps_min`: The minimum value of epsilon after decay.
- `learning_rate`: The learning rate for the optimizer.

Example training configuration:
```json
{
    "episodes": 1000,
    "batch_size": 32,
    "max_env_steps": 50,
    "batch_update": 20,
    "replay_buffer_size": 50000,
    "hidden_size": 64,
    "gamma": 0.99,
    "epsilon": 1,
    "eps_decay": 0.99,
    "eps_min": 0.01,
    "learning_rate": 0.001
}
```
