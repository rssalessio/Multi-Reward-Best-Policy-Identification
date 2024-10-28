from agents.dbmr_bpi.dbmr_bpi_config import DBMRBPIConfig
from agents.dbmr_bpi.dbmr_bpi import DBMRBPI
from agents.dqn.dqn_config import DQNConfig
from agents.rnd.rnd_config import RNDConfig
from agents.rnd.rnd import RNDAgent
from agents.disagreement.disagreement_config import DisagreementConfig
from agents.disagreement.disagreement import DisagreementAgent
from agents.apt.apt_config import APTConfig
from agents.apt.apt import APTAgent
from configs.config import EnvSimulationConfig, DeepSeaConfig


dqn_config = DQNConfig(
    hidden_layer_size=64, lr=5e-4, epsilon_0=10, target_soft_update=1e-2)

params = [(20, 50000, 25000), (30, 100000, 50000)]

PARAMETERS = [
    EnvSimulationConfig(
        collection_steps=steps,
        seeds=[i for i in range(30)],
        model_checkpoint_frequency=frequency,
        env_config=DeepSeaConfig(size=size, randomize=True, num_random_rewards=20),
        agents_config={
            DBMRBPI.NAME: DBMRBPIConfig(
                ensemble_size=20, prior_scale=8,  lr_Q=5e-4, lr_M=5e-5,
                epsilon_0=10, kbar=1, hidden_layer_size=64,
                mask_prob=0.5, noise_scale=0.0, delta_min=1e-8,
                target_soft_update=1e-2, exploration_prob=0.5, enable_mix=True, per_step_randomization=True
            ),

            RNDAgent.NAME: RNDConfig(
                            dqn_config=dqn_config,
                            rnd_rep_dim=512, rnd_scale=1, lr=1e-4
                        ),
            
            APTAgent.NAME: APTConfig(
                dqn_config=dqn_config,
                    rep_dim=512,
                    lr=1e-4,
                    hidden_layer_size=64
            ),
            DisagreementAgent.NAME: DisagreementConfig(
                dqn_config=dqn_config,
                ensemble_size=20, lr=1e-4, hidden_layer_size=64, 
            )
        }
    )
    for size, steps, frequency in params
]