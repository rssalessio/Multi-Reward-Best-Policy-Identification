import numpy as np
import numpy.typing as npt
from tabular.envs.mdp import MDP
from typing import Tuple

class NArms(MDP):
    """NArms environment

    This is an adaptation of the 6 arms environment
    @See also https://www.sciencedirect.com/science/article/pii/S0022000008000767
    An analysis of model-based Interval Estimation for Markov Decision Processes, Strehl & Littman, 2008

    Probability of transitioning from 0 to i is 1 if i=1, otherwise it's p0 / i
    Pr(i|i,a) = 1 if a >= i and Pr(0|i,a)=1 if a < 1
    """
    current_state: int                      # Current state
    length: int                             # Chain length
    p: float                                # Transition probability
    RIGHT = 0
    LEFT = 1
    
    def __init__(self, 
                 num_arms: int = 6,
                 p0: float = 1.0):
        """Initialize a double chain environment

        Parameters
        ----------
        num_arms : int, optional
            Number of arms, by default is 5
        p0: float, optional
            Transition probability coefficient, by default is 1
        """
        self.p0 = p0
        ns = num_arms + 1
        na = num_arms
    
        transitions = np.zeros((ns, na, ns))
        
        # Create transitions
        for s in range(num_arms):
            transitions[0, s, s+1] = 1 if s == 0 else self.p0 / (s + 1)
            transitions[0, s, 0] = 1 - transitions[0, s, s+1]

            transitions[s+1, s:, 0] = 1
            transitions[s+1, :s, s+1] = 1

        super().__init__(transitions)
        # Reset environment
        self.reset()

    
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
        assert 0 <= action <= self.dim_action, 'Action needs to either 0 or 1'
        next_state = np.random.choice(self.dim_state, p=self.P[self.current_state, action])
        rew = None if reward is None else reward[self.current_state, action]
        self.current_state = next_state
        return next_state, rew
    