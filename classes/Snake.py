import pygame

from classes.Config import Config

class Snake:
    def __init__(self, display):
        self.x_pos = (Config['game']['width'] -30) / 2
        self.y_pos = (Config['game']['height'] - 30) / 2
        self.display = display
        self.body = []
        self.max_size = 0

    def eat(self):
        self.max_size += 1

    def draw(self):
        return pygame.draw.rect(
            self.display, 
            Config['colors']['green'],
            [
                self.x_pos,
                self.y_pos,
                Config['snake']['height'],
                Config['snake']['width']
            ]
        )

    def draw_body(self):
        for item in self.body:
            pygame.draw.rect(
                self.display, 
                Config['colors']['green'],
                [
                    item[0],
                    item[1],
                    Config['snake']['width'],
                    Config['snake']['height']
                ]
            )

    def move(self, x_change, y_change):
        self.body.append((self.x_pos, self.y_pos))
        self.x_pos += x_change
        self.y_pos += y_change

        if len(self.body) > self.max_size:
            del(self.body[0])
