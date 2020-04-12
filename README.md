# RL-RL: Reinforcement Learning Rocket League Bot

## Reinforcement Learning agent for the game Rocket League.

This is an attempt to make a reinforcement learning agent for the game Rocket League: https://www.rocketleague.com/

This is done by using the RLBot framework: https://www.rlbot.org/

This project is being done for McGill's Comp767: Reinforcement Learning

The current goal is to teach various aspects of the game to an agent, in order to discover game mechanics.

The first objective is to learn the fastest way to reach the ball when a round starts, known as kickoff.

# Episodic Kickoffs
![](https://github.com/danielbairamian/RL-RL/blob/master/ReadmeMedia/episode.gif)

The episode for this task is defined as following:

- The bot always starts from the same kickoff position
- Once the episode starts, the bot will have 3 seconds to hit the ball
- The episode is done when either the bot hits the ball, or the timer runs out
- The agent will always receive a negative reward at every step of the episode
- The reward itself is scaled by the distance to the ball to incentivize rushing to the ball

## Episodic Consistency

In regular Rocket League games, the car can spawn in a finite set of multiple positions for kickoff.

Useful information about this can be found on the RLBot Wiki: https://github.com/RLBot/RLBot/wiki/Useful-Game-Values

Cars in Rocket League have a resource called "Boost", which allows them to go faster and fly.
When they spawn, cars start with 33 boost. They can either pickup small boost pads that give them +12 boost,
or they can pickup big boost pads that fills their boost to 100. 

As the RLBot API does not allow to reset boost pads on the floor, in order to make episodes consistent,
I took the decision to give the agent an additional +12 boost (as if he had taken a small boost pad), and
then disabled all boost pads. This has been done because all spawn positions have a small boost pad in front of them.

# Installation guide

## Game Requirement (RL & RLBot)

In order to run this project, obviously the first requirement is to own a legitimate copy of the Rocket League

Installing RLBot is not necessary for this project, as I have exposed the library and included in the project. This had to be
done to change the main loop logic to integrate learning.

## Python Requirements

RLBot supports many languages, and for obvious reasons I'm using Python.

I'm using the OpenAI spinningup library for the learning part of this, and most of the requirements are found there: 
https://spinningup.openai.com/en/latest/user/installation.html. Make sure the cloned repository is placed in the root of this project.

Simply follow the installation guide for openAI, and most requirements would be satisfied.

If you're on windows, your spinningup installation might fail because of MPI. I included the exe file in the DriverInstaller folder.
Simply run the msmpisetup.exe file and it should fix the error. Alternatively, you can download it from here: https://www.microsoft.com/en-us/download/details.aspx?id=57467 

An additional requirement is clr, which can be installed using "pip install pythonnet".

A requirement.txt file should be provided in the project for a simpler installation

Note, as spinningup supports both Tensorflow and Pytorch, feel free to use whichever, but make sure you specify
which one you're using, by changing the config file in "Spinningup/spinup/user_config.py"

## 3rd Party Requirements

The following application are optional, and are not necessary to run my Bot:


I included some 3rd party applications, for either quality or performance reasons:

- BakkesMod: This is a third party application that allows to run RocketLeague with rendering turned off.
This is useful if you want train the bot yourself. The exe file can be found here http://bakkesmod.com/. Althrough an additional step is required for this: After installing BakkesMod, find the config.cfg
file, line 68, set cl_rendering_disabled to "1".  If done correctly, this is how the game should look when running without rendering:

![](https://github.com/danielbairamian/RL-RL/blob/master/ReadmeMedia/rendering_disabled.png)

To enable this, go to the run.py file in the main folder, and set DISABLE_RENDERING to True

- GamePadViewer: This is a third party application that allows to visualize the bot's action.
I figured, what would be the point of making a bot that discovered game mechanics if we can't see them?
GamePadViewer launches a local web application that renders a controller, which is mapped to the Bot ingame.
The exe file is provided for this, however the ScpDriverInstaller.exe in the DriverInstaller folder must be run once.


When the game is running, go to https://gamepadviewer.com/ and you should see the following:

![](https://github.com/danielbairamian/RL-RL/blob/master/ReadmeMedia/controller.png)

To enable this, go to the run.py file in the main folder, and set CONTROLLER_VIZ to True.

Note: If you decide to run the visualizer, and you kill your simulation, the visualizer will not stop running.
As it emulates an XBox controller, it can interfere on your computer, and you must kill it.

In order to kill it, run the end_vis.bat file found at the root of the project.

# Running the Bot

Once everything is installed correctly, simply run the run.py file. Make sure Rocket League is closed before running,
as RLBot launched the game with specific flags. You should see the game open with the bot playing

# Reinforcement Learning and Results

tbd
