import logging
import torch
import numpy as np

# Classes and Functions for Logging
"""
Refer to https://docs.python.org/3/library/logging.html#logging.basicConfig and
 https://docs.python.org/3/library/logging.html#logrecord-attributes
 for more information on Formatter attributes, etc

-Logger: This is the class whose objects will be used in the application code directly to call the functions.
-LogRecord: Loggers automatically create LogRecord objects that have all the information related to the event being logged, like the name of the logger, the function, the line number, the message, and more.
-Handler: Handlers send the LogRecord to the required output destination, like the console or a file. Handler is a base for subclasses like StreamHandler, FileHandler, SMTPHandler, HTTPHandler, and more. These subclasses send the logging outputs to corresponding destinations, like sys.stdout or a disk file.
-Formatter: This is where you specify the format of the output by specifying a string format that lists out the attributes that the output should contain.
"""
class Logger:
    def __init__(self, name, handlers, log_file_name, level=logging.INFO):
        """
        Inputs:
            name (str): Name of the logger
            handlers (list(str)): specifies handler used for the logger. Possible options ['StreamHandler', 'FileHandler']
            log_file_name (str): if handlers include 'FileHandler' then this will be the name of the file that the logs will be saved in
            level = Logger level
        """
        for h in handlers:
            assert h in ['StreamHandler', 'FileHandler']
        
        self.ctx = {} # used to store values

        # Create a custom logger
        self.logger = logging.getLogger('MyCustomLogger')
        # Set Handler, Formatter, Level
        if 'StreamHandler' in handlers:
            s_handler = logging.StreamHandler() # writes logs to stdout (console)
            s_handler.setLevel(level)
            self.logger.addHandler(s_handler)
            s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            s_handler.setFormatter(s_format)
        if 'FileHandler' in handlers:
            f_handler = logging.FileHandler(log_file_name, mode='w') # writes logs to a file
            f_handler.setLevel(level)
            self.logger.addHandler(f_handler)
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)

    
    def store(self, **kwargs):

        """
        Save something into the Logger's current state.
        """
        for k, v in kwargs.items():
            container = self.ctx.get(k, [])
            container.append(v)
            self.ctx[k] = container
    
    def log_tensorboard(self, tb_summaryWriter, critic_model, index, gamma=0.9999): # TODO: set gamma and take this critic_model out of here to __init__
        """
        Log values to tensorboard.
        """
        state = self.ctx['state'] # [N, obs_dim]
        action = self.ctx['action']
        reward = self.ctx['reward']
        # Get Q(s,a) preds
        state_batch = torch.tensor(state)
        action_batch = torch.tensor(action)
        with torch.no_grad():
            Q_preds = critic_model(state_batch, action_batch).numpy()

        # Calculate Returns (G)
        G = []
        cur_G = 0
        for r in reversed(reward):
            cur_G = r + (gamma * cur_G)
            G.insert(0, cur_G.item()) 
        G = np.array(G)

        # Average results
        G = G.mean()
        Q_preds = Q_preds.mean()

        # Log to tensorboard
        tb_summaryWriter.add_scalar("Critic/discounted_return_mean", G, index)
        tb_summaryWriter.add_scalar("Critic/Q_value_mean", Q_preds, index)

        # Remove old episodes records
        del self.ctx['state']
        del self.ctx['action']
        del self.ctx['reward']


    def log_tabular(self, key, value=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        self.logger.info(f'{key} = {value}')
        # self.logger.warning('This is a warning')
        # self.logger.info('This is an info')
        # self.logger.error('This is an error')
        pass

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        pass

# logger = Logger("MyCustomLogger", ['StreamHandler', 'FileHandler'], 'file.log')


