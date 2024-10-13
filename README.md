# Serverless optimization using deep reinforcement learning
## Code structure
- `env/` folder: Implementation of serverless environment base-on Gymnasium library
- `utils/` folder which include:
    - `models.py`, `dqn_agent.py`: Neural network and Deep Q-learning implementation. Most of source code in these two files comes from https://github.com/MOSAIC-LAB-AALTO/drl_based_trigger_selection
    - Others function to log, debug and support for main function
- `main_dqn.py`: Include main function for training and testing DRL agorithms with serverless environment.
## How to run code?
So far we have only deployed the DQN algorithm for the purpose of testing the environment's behavior. 
To train model, run command:
```
python main_dqn.py -m dqn -p hyperparameters.txt 
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
        ├── hyperparameters.txt
        ├── profiling.npz
        └── train/
            ├── live_average_rewards_DQN.png
            └── log.txt
```
Trained model's weight is saved in `dqn_network_` and `target_dqn_network_`. System's information and reward of each episode in training process are saved in `log.txt`. Random resource profiling array is saved in `profiling.npz`. Env and model parameters is saved in `hyperparameters.txt`.

You can also open `live_average_rewards_DQN.png` by vscode to monitor the training process. When the training process is complete, the program will also automatically test model using the saved weights. Testing results is saved in `result/test/` folder.

You can also use saved model's weights of other training times by running following command (replace `n` with number episode you want to test and  `your_main_folder/result/result_{id}` with folder which contains model's weights file).
```
python main_dqn.py -m dqn -p "hyperparameter_json_file" -o n -f "your_main_folder/result/result_{id}"
```
