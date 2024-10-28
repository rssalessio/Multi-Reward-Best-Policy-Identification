from agents.dbmr_bpi.dbmr_bpi_config import DBMRBPIConfig
from agents.dbmr_bpi.dbmr_bpi import DBMRBPI
from agents.dqn.dqn_config import DQNConfig
from agents.rnd.rnd_config import RNDConfig
from agents.rnd.rnd import RNDAgent
from agents.disagreement.disagreement_config import DisagreementConfig
from agents.disagreement.disagreement import DisagreementAgent
from agents.apt.apt_config import APTConfig
from agents.apt.apt import APTAgent

from configs.config import EnvSimulationConfig, CartpoleSwingupConfig


params = [(3, 150000, 50000), (5, 200000, 50000)]

dqn_config = {
    num_steps: DQNConfig(
    replay_capacity=num_steps,
    hidden_layer_size=128, lr=5e-4, epsilon_0=10, target_soft_update=1e-2)
    for _, num_steps, _ in params
    }


PARAMETERS = [
    EnvSimulationConfig(
        collection_steps=steps,
        model_checkpoint_frequency=frequency,
        env_config = CartpoleSwingupConfig(
            height_threshold= k / 20, x_reward_threshold= 1 - k/20,
            num_base_rewards = 5,
            num_random_rewards = 5
        ),
        seeds=[i for i in range(30)],
        agents_config={
            DBMRBPI.NAME: DBMRBPIConfig(
                replay_capacity=steps,
                ensemble_size=20, prior_scale=6, 
                lr_Q=5e-4, lr_M=5e-4, target_soft_update=1e-2,
                epsilon_0=10, kbar=2, hidden_layer_size=128,
                mask_prob=0.7, noise_scale=0.0, delta_min=1e-2,
                exploration_prob=0.5, enable_mix=True, per_step_randomization=True
            ),

            RNDAgent.NAME: RNDConfig(
                dqn_config=dqn_config[steps],
                rnd_rep_dim=512, rnd_scale=1, lr=1e-4
            ),
            
             APTAgent.NAME: APTConfig(
                dqn_config=dqn_config[steps],
                    rep_dim=512,
                    lr=1e-4,
                    hidden_layer_size=128
            ),
            DisagreementAgent.NAME: DisagreementConfig(
                dqn_config=dqn_config[steps],
                ensemble_size=20, lr=1e-4, hidden_layer_size=128, 
            ),  
        }
    )
    for k, steps, frequency in params
]