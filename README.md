# Serverless optimization using deep reinforcement learning
## Code structure
- rlss_env.py: Simulate serverless environment base-on Gymnasium library
- models.py, dqn_agent.py: Include Neural network and Deep Q-learning . Most of source code of these two files comes from https://github.com/MOSAIC-LAB-AALTO/drl_based_trigger_selection
- main_dqn.py: Include main function for training and testing DRL agorithms with serverless environment.
- utils: others function to log, debug and support for main function
## How to run code?
So far we have only deployed the DQN algorithm for the purpose of testing the environment's behavior. 
To train model, run command:
```
python main_dqn.py --train dqn
```
After run following command, program will create result/ folder with following structure
```
your_main_folder/
├── {other files}
├── main_dqn.py
├── result/
    ├── result_{id}/
        ├── dqn_network_
        ├── target_dqn_network_
        └── train/
            ├── live_average_rewards_DQN.png
            └── log.txt
```
"dqn_network_" and "target_dqn_network_" contain model's weight. "log.txt" contain system's information and training logging for each step ad episode. You can also open "live_average_rewards_DQN.png" by vscode to monitor the training process. When the training process is complete, the program will also automatically test model using the saved weights. Testing's result is saved in result_{id}/test folder.

You can also use saved model's weights of other training times by running following command, replace "n" with number episode you want to test and "your_main_folder/result/result_{id}" with your folder which contains model's weights file.
```
 python main_dqn.py --train dqn --observe n --folder "your_main_folder/result/result_{id}"
```
