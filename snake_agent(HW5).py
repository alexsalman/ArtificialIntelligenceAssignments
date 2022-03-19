# Alex Salman - aalsalma@ucsc.edu
# 3/18
# CSE240 Winter 2022

import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr)
    #   gamma which is another parameter helpful in calculating next move, in other words
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write.
    #   Function Helper:IT gets the current state, and based on the
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on.
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        print("IN helper_func")
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]

        # snake_head_x
        if snake_head_x <= helper.BOARD_LIMIT_MIN:
            snake_head_x_prime = 1
        elif snake_head_x >= helper.BOARD_LIMIT_MAX:
            snake_head_x_prime = 2
        else:
            snake_head_x_prime = 0

        # snake_head_y
        if snake_head_y <= helper.BOARD_LIMIT_MIN:
            snake_head_y_prime = 1
        elif snake_head_y >= helper.BOARD_LIMIT_MAX:
            snake_head_y_prime = 2
        else:
            snake_head_y_prime = 0

        # (0,0) case
        if snake_head_x <= 0 or snake_head_y <= 0 or snake_head_x >= helper.IN_WALL_COORD or snake_head_y >= helper.IN_WALL_COORD:
            snake_head_x_prime, snake_head_y_prime = 0, 0

        # food_x
        if snake_head_x > food_x:
            food_x_prime = 1
        elif snake_head_x < food_x:
            food_x_prime = 2
        else:
            food_x_prime = 0

        # food_y
        if snake_head_y > food_y:
            food_y_prime = 1
        elif snake_head_y < food_y:
            food_y_prime = 2
        else:
            food_y_prime = 0

        # snake_body

        # body in top
        if (snake_head_x, snake_head_y - helper.BOARD_LIMIT_MIN) in snake_body:
            body_top = 1
        else:
            body_top = 0
        # body in bottom
        if (snake_head_x, snake_head_y + helper.BOARD_LIMIT_MIN) in snake_body:
            body_bottom = 1
        else:
            body_bottom = 0
        # body in left
        if (snake_head_x - helper.BOARD_LIMIT_MIN, snake_head_y) in snake_body:
            body_left = 1
        else:
            body_left = 0
        # body in right
        if (snake_head_x + helper.BOARD_LIMIT_MIN, snake_head_y) in snake_body:
            body_right = 1
        else:
            body_right = 0

        state_prime = (snake_head_x_prime, snake_head_y_prime, food_x_prime, food_y_prime, body_top, body_bottom, body_left, body_right)

        return state_prime


    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write.
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make
    #   using the compute reward function defined above.
    #   This function also keeps track of the fact that we are in
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make.
    #   the LPC variable can be used to determine the learning rate (lr), but if
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively.
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):
        print("IN AGENT_ACTION")
        state_prime = self.helper_func(state)
        action = self.a
        # compute reward
        reward = self.compute_reward(points, dead)
        if self._train:
            if self.s != None:
                q_prime = np.max(self.Q[state_prime])
                alpha = self.LPC / (self.LPC + self.N[self.s + (action,)])
                q = self.Q[self.s + (action,)]
                self.Q[self.s + (action,)] += alpha * (reward + self.gamma * q_prime - q)

            # action with exploration
            max_value = float('-inf')
            required_action = 0
            for i in range(3,-1,-1):
                if self.N[state_prime + (i,)] < self.Ne:
                    val = 1
                else:
                    val = self.Q[state_prime + (i,)]

                if val > max_value:
                    max_value = val
                    required_action = i

            if not dead:
                self.N[state_prime + (required_action,)] += 1
                self.points = points
            self.s = state_prime
            self.a = required_action
        else:
            required_action = np.argmax(self.Q[state_prime])

        if dead:
            self.reset()

        # RETURN THE REQUIRED ACTION.
        return required_action
