import gym
import numpy as np
import random
import pygame
from gym import spaces

PIPE_WIDTH = 52
BIRD_SIZE = 24

class FlappyBirdEnv(gym.Env):
    """
        Flappy Bird Environment với pygame render
        State: [bird_y, bird_vel, pipe_dist_to_bird, pipe_top_y]
        Actions: 0 = do nothing, 1 = flap
        Reward: +1 mỗi frame sống, -100 khi chết, +5 khi qua ống
    """
    def __init__(self, width=288, height=512, pipe_gap=120, pipe_speed=3, render_mode=False):
        super().__init__()
        self.W = width
        self.H = height
        self.pipe_gap = pipe_gap
        self.pipe_speed = pipe_speed

        # action & state space
        self.action_space = spaces.Discrete(2)
        low = np.array([0.0, -20.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([float(self.H), 20.0, float(self.W), float(self.H)], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # vị trí chim
        self.bird_X = 50

        # render flag
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Flappy Bird RL")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.bird_y = self.H // 2
        self.bird_vel = 0.0
        self.gravity = 0.8
        self.flap_power = -9.0

        self.pipe_x = self.W
        self.pipe_top = self._sample_pipe()

        self.score = 0
        self.done = False
        return self.__get_state()

    def _sample_pipe(self):
        return random.randint(50, self.H - 50 - self.pipe_gap)

    def __get_state(self):
        dist = max(0.0, self.pipe_x - self.bird_X)
        return np.array([self.bird_y, self.bird_vel, dist, float(self.pipe_top)], dtype=np.float32)

    def step(self, action):
        if action == 1:
            self.bird_vel = self.flap_power

        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        # di chuyển ống
        self.pipe_x -= self.pipe_speed
        reward = 1.0

        if self.pipe_x + PIPE_WIDTH < self.bird_X and not hasattr(self, "passed"):
            self.passed = True
            reward += 5.0
            self.score += 1

        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = self.W
            self.pipe_top = self._sample_pipe()
            if hasattr(self, "passed"):
                delattr(self, "passed")

        done = False
        if self.bird_y > self.H or self.bird_y < 0:
            done = True
            reward = -100.0

        # va chạm ống
        pipe_bottom = self.pipe_top + self.pipe_gap
        if (self.bird_X + BIRD_SIZE > self.pipe_x and self.bird_X < self.pipe_x + PIPE_WIDTH):
            if self.bird_y < self.pipe_top or self.bird_y + BIRD_SIZE > pipe_bottom:
                done = True
                reward = -100.0

        self.done = done
        return self.__get_state(), reward, done, {"score": self.score}

    def render(self, mode="human"):
        if not self.render_mode:
            print(f"bird_y={self.bird_y:.1f}, vel={self.bird_vel:.1f}, pipe_x={self.pipe_x}, pipe_top={self.pipe_top}")
            return

        self.screen.fill((135, 206, 235))  # sky blue

        # pipes
        pygame.draw.rect(self.screen, (0, 200, 0), (self.pipe_x, 0, PIPE_WIDTH, self.pipe_top))
        pygame.draw.rect(self.screen, (0, 200, 0), (self.pipe_x, self.pipe_top + self.pipe_gap, PIPE_WIDTH, self.H))

        # bird
        pygame.draw.rect(self.screen, (255, 255, 0), (self.bird_X, int(self.bird_y), BIRD_SIZE, BIRD_SIZE))

        # score text
        font = pygame.font.SysFont(None, 36)
        score_surface = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_mode:
            pygame.quit()
