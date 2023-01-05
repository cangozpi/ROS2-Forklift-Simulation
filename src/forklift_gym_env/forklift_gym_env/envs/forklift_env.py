import gym
from gym import spaces
import numpy as np
from forklift_gym_env.envs.utils import generateLaunchDescriptionForkliftEnv, startLaunchServiceProcess

class ForkliftEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], # TODO: set this to supported types
        "render_fps": 4 #TODO: set this
    }

    def __init__(self, render_mode = None):
       # set types of observation_space and action_space 
       self.observation_space = spaces.Dict({
        "agent": spaces.Box(low = 0, high = 1, shape=(2, ),  dtype=int),
        "target": spaces.Box(low = 0, high = 1, shape=(2, ),  dtype=int)
       })
       self.action_space = spaces.Discrete(4) # TODO: change this

       # set render_mode
       assert render_mode is None or render_mode in self.metadata["render_modes"]
       self.render_mode = render_mode

       # self.clock` will be a clock that is used to ensure that the environment is rendered at the correct framerate in human-mode.
       self.clock = None

       # start gazebo simulation, spawn forklift model, start controllers
       launch_desc = generateLaunchDescriptionForkliftEnv() # generate launch description
       self.launch_subp = startLaunchServiceProcess(launch_desc)# start the generated launch description on a subprocess
    

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}


    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int) # TODO: set this to agent (forklift) location at start in gazebo

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human": # TODO: handle this for rendering gazebo simulation
            self._render_frame()

        return observation, info

    
    def step(self, action):
        self._agent_location = self._agent_location + 1 # TODO: update this with the new location read from ros subscribers to the state of the agent
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # TODO: change this once the reward function is figured out
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human": # TODO: handle this once the simulation is figured out with gazebo
            self._render_frame()

        return observation, reward, terminated, False, info # (observation, reward, done, truncated, info)


    def render(self): # TODO: rewrite for gazebo
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self): # TODO: rewrite for gazebo
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


    def close(self): # TODO: close any resources that are open (e.g. ros2 nodes, gazebo, rviz e.t.c)
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()