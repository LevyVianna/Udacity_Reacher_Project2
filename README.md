# Reacher
[image1]: https://github.com/LevyVianna/Udacity_Reacher_Project2/blob/main/reacher.gif?raw=true "Trained Agents"

### Game Rules
![Traineds Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes. In this case, using 20 agents, they should get an average score of +30 (100+ consecutive episodes and over all agents).

### Getting Started
1. I recommend you install python 3.6 and the virtual environment you prefer.
2. You need to download the Unity environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
   You will need to use this file when you run the "Solution" notebook
 3. Run "jupyter notebook" and click in "Solution.ipynb" notebook.
 4. You need to run the first cell from "Solution.ipynb" notebook - with the command "!pip -q install ./python" - to install all python dependencies.
 5. In the "Solution.ipynb" notebook, in step 3, put the path of the Unity environment file downloaded.
