# This file contains the parameters used in the simulations

from typing import NamedTuple, List
from tabular.simulation_parameters import SimulationConfiguration, SimulationGeneralParameters, EnvType, SimulationParameters, EnvParameters, DoubleChainParameters, RiverSwimParameters, make_env, NArmsParameters, ForkedRiverSwimParameters
from tabular.agents.mr_nas import MRNaSParameters
from tabular.agents.random import RandomAgentParameters
from tabular.agents.ide3al import IDE3ALParameters
from tabular.agents.rf_ucrl import RFUCRLParameters
from tabular.agents.mr_psrl import MRPSRLparameters
from tabular.utils.period import ConstantPeriod



CONFIG = SimulationConfiguration(
    sim_parameters=SimulationGeneralParameters(
        num_sims=100,
        num_rewards=30,
        freq_eval=500,
        discount_factor=0.9,
        delta=1e-2
    ),
    envs = [
    (
      EnvParameters(EnvType.FORKED_RIVERSWIM, ForkedRiverSwimParameters(river_length=4), 50000),
      [ 
        IDE3ALParameters(agent_parameters=None, xi=None),
        MRPSRLparameters(agent_parameters=None),
        RFUCRLParameters(
            agent_parameters=None
        ),
        MRNaSParameters(agent_parameters = None,
                        enable_averaging=True,
                        alpha=0.99,
                        beta=0.01,
                        period_computation_omega=ConstantPeriod(250)
        )
      ]
     ),
     (
      EnvParameters(EnvType.RIVERSWIM, RiverSwimParameters(num_states=10), 50000),
      [ 
        IDE3ALParameters(agent_parameters=None, xi=None),
        MRPSRLparameters(agent_parameters=None),
        RFUCRLParameters(
            agent_parameters=None
        ),
        MRNaSParameters(agent_parameters = None,
                        enable_averaging=True,
                        alpha=0.99,
                        beta=0.01,
                        period_computation_omega=ConstantPeriod(100)
        )
      ]
     ),
     (
      EnvParameters(EnvType.N_ARMS, NArmsParameters(num_arms=4,p0=1), 50000),
      [ 
        IDE3ALParameters(agent_parameters=None, xi=None),
        MRPSRLparameters(agent_parameters=None),
        RFUCRLParameters(
            agent_parameters=None
        ),
        MRNaSParameters(agent_parameters = None,
                        enable_averaging=True,
                        alpha=0.99,
                        beta=0.01,
                        period_computation_omega=ConstantPeriod(10000)
        )
      ]
     ),
     (
      EnvParameters(EnvType.DOUBLE_CHAIN, DoubleChainParameters(length=6), 50000),
      [ 
        IDE3ALParameters(agent_parameters=None, xi=None),
        MRPSRLparameters(agent_parameters=None),
        RFUCRLParameters(
            agent_parameters=None
        ),
        MRNaSParameters(agent_parameters = None,
                        enable_averaging=True,
                        alpha=0.99,
                        beta=0.01,
                        period_computation_omega=ConstantPeriod(100)
        )
      ]
     )
    ],
)
