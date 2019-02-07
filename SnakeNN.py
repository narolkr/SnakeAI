import pygame

from classes.Game import Game
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
from classes.Config import Config


class SnakeNN:
    def __init__(self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'snake_nn_2.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            display = pygame.display.set_mode((
                Config['game']['width'], 
                Config['game']['height']
            ))
            pygame.display.set_caption(Config['game']['caption'])

            game = Game(display)
            
            _, prev_score, snake, apple = game.start()
            prev_observation = self.generate_observation(snake, apple)
            prev_apple_distance = self.get_apple_distance(snake, apple)
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, apple  = game.step(game_action)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    apple_distance = self.get_apple_distance(snake, apple)
                    if score > prev_score or apple_distance < prev_apple_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, apple)
                    prev_apple_distance = apple_distance
        return training_data

    def generate_action(self, snake):
        action = randint(0,2) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def generate_observation(self, snake, apple):
        snake_direction = self.get_snake_direction_vector(snake)
        apple_direction = self.get_apple_direction_vector(snake, apple)
        # barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        # barrier_front = self.is_direction_blocked(snake, snake_direction)
        # barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, apple_direction)
        return np.array([angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        if snake.y_pos-snake.prev_y_pos == 0:
            return np.array([snake.x_pos,0])
        else:
            return np.array([0,snake.y_pos])

    def get_apple_direction_vector(self, snake, apple):
        pos = np.array([apple.x_pos - snake.x_pos , apple.y_pos - snake.y_pos])
        return pos

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_apple_distance(self, snake, apple):
        return np.linalg.norm(self.get_apple_direction_vector(snake, apple))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = Game()
            _, score, snake, apple = game.start()
            prev_observation = self.generate_observation(snake, apple)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, apple  = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print(steps)
                    print(snake)
                    print(apple)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, apple)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):
        game = SnakeGame(gui = True)
        _, _, snake, apple = game.start()
        prev_observation = self.generate_observation(snake, apple)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, apple  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, apple)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)
        self.visualise_game(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    SnakeNN().train()
