# USING REINFORMENT LEARNING TO PLAY KUNG FU MASTER 
The environment chosen for the project is Kung Fu Master, a game by Atari in which you play as a Kung-Fu master who is on his way to a throne of evil magicians to rescue the public. Princess Victoria, you will have to defeat the enemies on the way there to be able to rescue the princess.
The Kung Fu Master environment will consist of 14 actions:
* 0 -	NOOP
* 1 -	UP
* 2 -	RIGHT
* 3 -	LEFT
* 4 -	DOWN
* 5 - 	DOWNRIGHT
* 6 -	DOWNLEFT
* 7 -	RIGHTFIRE
* 8 -	LEFTFIRE
* 9 -	DOWNFIRE
* 10 -	UPRIGHTFIRE
* 11 -	UPLEFTFIRE
* 12 -	DOWNRIGHTFIRE
* 13 -	DOWNLEFTFIRE
  
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/b44a8137-2386-4a2c-864d-6925b0eaeab8.png" alt="drawing" width="25%" height="25%"/>
<br>
<a style="text-align: center">Observation </a>
</p>

# Preprocess and backbone
Each observation obtained from the first Kung Fu Master game will be converted to grayscale image and resized to size (84, 84)

<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/4d44190d-7ef3-4552-b836-09f002f21939.png" alt="drawing" width="50%" height="50%"/>
<br>
<a style="text-align: center">Backbone </a>
</p>

# Methods
## DEEP Q NETWORK (DQN)
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/1318a981-4fc9-46ea-8393-f4784350b933.png" alt="drawing" width="50%" height="50%"/>
</p>

<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/8d98242d-b0a0-4aab-9b92-1f57d22a2405.png" alt="drawing" width="50%" height="50%"/>
</p>

The Q-Network model parameter will be updated as follows:
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/cdca9d4f-766d-4b4c-ba9c-284048f2e3fe.png" alt="drawing" width="50%" height="50%"/>
</p>

## DOUBLE DEEP Q NETWORK (DDQN)
Different from DQN, $Y_t$ will be calculated as follows
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/3d9b5832-0ae1-4dd9-8abd-487c9b0ebf47.png" alt="drawing" width="50%" height="50%"/>
</p>

## PRIORITIZED DDQN
Prioritized Experience Replay (PER) was first introduced in 2015 by Tom Schaul. The idea of PER is that in experiences stored in memory, there will be some experiences that are important for training, and these experiences may appear less often. Like the normal Experience Replay, we will randomly choose the experience until there is enough 1 batch, which makes it even more difficult for important experiences to be selected for training.

PER uses temporal-difference (TD) error to calculate priority for each experience so that the probability that important experiences can be used for training is higher (pure greedy prioritization), thereby helping the training process to happen. better.

## DUELING DQN
Dueling DQN was first introduced by the Google DeepMind research team, which included Tom Schaul and van Hasselt. Instead of using a 1-thread Q-network, we use a 2-thread Q-network to estimate the V(s) value for each state and the A(s,a) benefit value for each action pair – status. Then the two values V(s) and A(s,a) are combined to estimate Q(s,a). The purpose of the above approach is so that the Q-network can learn which states have (or do not) make sense in training without learning the effect of each action on a state.
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/b3bdead6-00e9-4482-9759-4c79d899e490.png" alt="drawing" width="50%" height="50%"/>
</p>

## CATEGORICAL DQN
Categorical DQN was introduced in 2017 by Marc G. Bellemare et al. The main goal of the algorithm is to learn the distribution of future rewards (return) instead of approximating their expected value like previous algorithms. With complex environments where rewards can be random, learning the expected value of the reward ignores the information needed to be able to choose the best course of action.

# Result
<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/a1e14628-9322-44ac-89c9-a484a4963234.png" alt="drawing" width="50%" height="50%"/>
<br>
<a style="text-align: center">Result with 200000 steps training </a>
</p>


<p align="center">
<img src="https://github.com/ptatien0307/reinforcement_learning/assets/79583501/9edd0a6f-83eb-4f3e-b574-6f3b68fd6714.png" alt="drawing" width="50%" height="50%"/>
<br>
<a style="text-align: center">Result with 500000 steps training </a>
</p>


