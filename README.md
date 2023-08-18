A repo to store reinforcement learning course

# Sokoban
* Using DFS, BFS, UCS to solve sokoban

<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/86b80e53-b366-44b0-bea1-74078d53c6c2.png" alt="drawing" width="20%" height="20%"/>
</p>

# Pacman
* Using minimax, alphabeta, expectimax to solve pacman
* Features for evaluation function
  * f1: Current game score
  * f2: Manhatten distance between Pacmand and the closest capsule (weight: -6)
  * f3: Manhattan distance between Pacman and the closest food (weight: -3)
  * f4: Manhattan distance between Pacman and the closest ghost (weight: -80 * inverse(f4))
  * f5: Manhattan distance between Pacman and the closest scared ghost (weight: -4)
  * f6: The number of remained foods – weight (weight: -15)
  * f7: The number of remained capsules – weight (weight: -30)
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/e703a080-1c0f-48d4-b73f-d7aea2c29947.png" alt="drawing" width="20%" height="20%"/>
</p>

# Policy evalution, value iteration, policy iteration
Using **gym library** (https://www.gymlibrary.dev/index.html) to create environment 
* FrozenLake4x4
* Taxi-v3

<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/dc2caaa6-ef10-4539-90dc-c42ea73badb0.png" alt="drawing" width="20%" height="20%"/>
</p>

<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/dbb9d0f2-9eea-43b5-8cc1-a05dc7ee3c09.png" alt="drawing" width="20%" height="20%"/>
</p>

# DQN, SARSA
Using **gym library** (https://www.gymlibrary.dev/index.html) to create environment 
* FrozenLake4x4
* Taxi-v3
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/ba40c68b-27fe-4e19-baa6-7250fd95b9a0.png" alt="drawing" width="20%" height="20%"/>
</p>

# Deep Q Network
Using **gym library** (https://www.gymlibrary.dev/index.html) to create environment 
Using Deep Q network to solve these 2 games

* CartPole-v0

<p align="center">
<img src="https://www.gymlibrary.dev/_images/cart_pole.gif" alt="drawing" width="20%" height="20%"/>
</p>

* MountainCar-v0
<p align="center">
<img src="https://www.gymlibrary.dev/_images/mountain_car.gif" alt="drawing" width="20%" height="20%"/>
</p>
