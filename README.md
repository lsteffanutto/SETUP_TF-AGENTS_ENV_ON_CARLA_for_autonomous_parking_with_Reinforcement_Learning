# PoC of KARlab UC3 on the Carla simulator

## Notes for beginners
- Tutorials for Carla simulator setup and tf-agents setup are avalaible at the bottom of the link below 
- The Carla modelisation of the UC3, advancement and explanation of this code are avalaible here [here](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2664529933/Mod+lisation+pratique+et+impl+mentation+du+Stationnement+Autonome+sur+Carla)
- In addition, to have other example projects, lots of souces are avalaible [here](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2581594131/Use+the+Unity+RL+Car+Parking+PoC+on+Carla)
- to launch a script
- See the comments of the issues "Voir comment intégrer ou reprendre le modèle Unity sur Carla" and "Recréer le modèle de RL Unity sur Carla"

## Scripts
- "hellocarla.py" is for the first tests on Carla for data collections and creation of the Parking environment. It is a remake of Carla script "tutorial.py" with an adaptation to the KARlab UC3
- "parking_env.py" is the creation of the Parking environment class in Carla simulator to train it with tf-agents
- "utils_carla.py" are all the functions used in Carla simulator
- "test_py_env.py" is a tutorial/example of tf-agents for the first test and comprehension of the tf-agents library 
- "train.py" is the RL loop and training of the agent with tf-agents
- "ppo.py" is use to compute the PPO RL algorithm

## To launch a script
- Activate the python environment with the necessary libraries of this project
- Clone this repo in:  " C:\Users\...\path_where_is_carla\PythonAPI\examples "
- Open an Anaconda prompt and go to the same path as above
- Launch Carla: " CarlaUE4.exe /Game/Carla/Maps/Town05 -quality-level=Low -ResX=800 -ResY=500 -fps=60 -windowed "
- Launch a script: " python ppo_carla\hellocarla.py " or " python ppo_carla\train.py "