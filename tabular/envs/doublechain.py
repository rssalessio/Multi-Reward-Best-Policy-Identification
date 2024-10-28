import numpy as np
import numpy.typing as npt
from tabular.envs.mdp import MDP
from typing import Tuple

class DoubleChain(MDP):
    """DoubleChain environment
    @See also https://arxiv.org/pdf/2102.12769.pdf
    """
    current_state: int                      # Current state
    length: int                             # Chain length
    p: float                                # Transition probability
    RIGHT = 0
    LEFT = 1
    
    def __init__(self, 
                 length: int = 5,
                 p: float = 0.7):
        """Initialize a double chain environment

        Parameters
        ----------
        length : int, optional
            Chain length, by default is 5
        p: float, optional
            Transition probability, by default is 0.7
        """
        self.length = length
        self.p = p
        ns = 2*length + 1
        na = 2
    
        transitions = np.zeros((ns, na, ns))
        s0, s1, sl, sl_1, s_2l_1 = 0, 1, length, length+1, ns-1
        
        # Create transitions
        for start, end in [(s1, sl), (sl_1, s_2l_1)]:
            for s in range(start, end+1):
                next_state = s+1 if s != end else end
                prev_state = s-1 if s != sl_1 else s0
                transitions[s, DoubleChain.RIGHT, next_state] = p
                transitions[s, DoubleChain.RIGHT, prev_state] = 1-p
                transitions[s, DoubleChain.LEFT, prev_state] = 1

        transitions[s0, DoubleChain.RIGHT, sl_1] = 1
        transitions[s0, DoubleChain.LEFT, s1] = 1

        super().__init__(transitions)
        # Reset environment
        self.reset()

    def build_reward_matrix(self, min_reward: float = 0.05, max_reward: float = 1.0):
        rewards = np.zeros((self.dim_state, self.dim_action))
        
        # Create rewards
        rewards[0, :] = min_reward
        rewards[self.length, DoubleChain.RIGHT] = max_reward
        rewards[self.dim_state - 1, DoubleChain.RIGHT] = max_reward
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
    
