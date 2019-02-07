import pygame 

from classes.Config import Config
from classes.Snake import Snake
from classes.Apple import Apple

class Game:
    def __init__(self, display):
        self.display = display
        self.score = 0
        self.done = False
        self.snake = Snake(self.display)
        self.apple = Apple(self.display)        
    
    def start(self):
        return self.generate_observations()

    
    def loop(self):
        clock = pygame.time.Clock()
        # self.snake = Snake(self.display)
        # self.apple = Apple(self.display)

        x_change = 0
        y_change = 0
        
        self.score = 0


        while True:
            for event in pygame.event.get():

                if self.done == True or event.type == pygame.QUIT:
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        x_change = -Config['snake']['speed']
                        y_change = 0
                    elif event.key == pygame.K_RIGHT:
                        x_change = Config['snake']['speed']
                        y_change = 0
                    elif event.key == pygame.K_UP:
                        x_change = 0
                        y_change = -Config['snake']['speed']
                    elif event.key == pygame.K_DOWN:
                        x_change = 0
                        y_change = Config['snake']['speed']
        
            # Fill background and draw game area
            self.display.fill(Config['colors']['green'])
            
            pygame.draw.rect(
                self.display, 
                Config['colors']['black'],
                [
                    0,
                    Config['game']['bumper_size'],
                    Config['game']['height'],
                    Config['game']['width']
                ]
            )
            
            # Draw an apple
            apple_rect = self.apple.draw()

            # Move and Re-Draw Snake
            self.snake.move(x_change, y_change)
            self.snake_rect = self.snake.draw()
            self.snake.draw_body()

            # Detect with corners
            if (self.snake.x_pos >= Config['game']['width'] and x_change>0):
                self.snake.x_pos = 0
            if (self.snake.x_pos < 0 and x_change<0):
                self.snake.x_pos = Config['game']['width']- 1
            if (self.snake.y_pos > Config['game']['height'] + Config['game']['bumper_size'] and y_change>0):
                self.snake.y_pos = Config['game']['bumper_size']
            if (self.snake.y_pos < Config['game']['bumper_size'] and y_change<0):
                self.snake.y_pos = Config['game']['height'] + Config['game']['bumper_size'] - 1

            # Detect collision with apple
            if apple_rect.colliderect(snake_rect):
                self.apple.randomize()
                self.score += 1
                self.snake.eat()

            # Collide with Self
            if len(self.snake.body) >= 1:
                for cell in self.snake.body:
                    if self.snake.x_pos == cell[0] and self.snake.y_pos == cell[1]:
                        self.done = True
                        # self.loop()

            # Initialize font and draw title and score text
            pygame.font.init()
            font = pygame.font.Font('./assets/Now-Regular.otf', 28)
            
            score_text = 'Score: {}'.format(self.score)
            score = font.render(score_text, True, Config['colors']['white'])
            title = font.render(Config['game']['caption'], True, Config['colors']['white'])


            title_rect = title.get_rect(
                center=(
                    60, 
                    Config['game']['bumper_size'] / 2
                )
            )

            score_rect = score.get_rect(
                center=(
                    Config['game']['width']/2, 
                    Config['game']['bumper_size'] / 2
                )
            )

            self.display.blit(score, score_rect)
            self.display.blit(title, title_rect)

            pygame.display.update()
            clock.tick(Config['game']['fps'])

    def generate_observations(self):
        return self.done, self.score, self.snake, self.apple
  