import numpy as np
import numpy.typing as npt
from tabular.envs.mdp import MDP
from typing import Tuple, NamedTuple


class RiverSwim(MDP):
    """RiverSwim environment
    @See also https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1374179
    """
    current_state: int                      # Current state
    
    def __init__(self, 
                 num_states: int = 5):
        """Initialize a river swim environment

        Parameters
        ----------
        num_states : int, optional
            Maximum number of states, by default 5
        """        
        ns = num_states
        na = 2
    
        transitions = np.zeros((ns, na, ns))
        
        # Create transitions
        for s in range(1, ns-1):
            transitions[s, 1, s] = 0.6
            transitions[s, 1, s-1] = 0.1
            transitions[s, 1, s+1] = 0.3
        transitions[1:-1, 0, 0:-2] = np.eye(num_states-2)

        transitions[0, 0, 0] = 1
        transitions[0, 1, 0] = 0.7
        transitions[0, 1, 1] = 0.3
        transitions[-1, 1, -1] = 0.3
        transitions[-1, 1, -2] = 0.7
        transitions[-1, 0, -2] = 1
        
        super().__init__(transitions)
        # Reset environment
        self.reset()

    def build_reward_matrix(self, min_reward: float = 0.05, max_reward: float = 1.0):
        rewards = np.zeros((self.dim_state, self.dim_action))
        
        # Create rewards
        rewards[0, 0] = min_reward
        rewards[-1, 1] = max_reward
        return rewards
    
    def reset(self) -> int:
        """Reset the current state

        Returns
        -------
        int
            initial state 0
        """        
        self.current_state = 0
        return self.current_state
    
    def step(self, action: int, reward: npt.NDArray[np.float64] | None = None) -> Tuple[int, float]:
        """Take a step in the environment

        Parameters
        ----------
        action : int
            An action (0 or 1)

        Returns
        -------
        Tuple[int, float]
            Next state and reward
        """        
        assert action == 0 or action == 1, 'Action needs to either 0 or 1'
        
        next_state = np.random.choice(self.dim_state, p=self.P[self.current_state, action])
        rew = None if reward is None else reward[self.current_state, action]
        self.current_state = next_state
        return next_state, rew
    