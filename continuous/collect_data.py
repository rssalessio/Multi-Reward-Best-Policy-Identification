import os
import numpy as np
from numpy.typing import NDArray
from agents.agent import Agent
from agents.dbmr_bpi.dbmr_bpi import DBMRBPI
from agents.rnd.rnd import RNDAgent
from agents.disagreement.disagreement import DisagreementAgent
from agents.apt.apt import APTAgent
from envs.deepsea import DeepSea
from typing import Callable, Union, Tuple, Dict, Callable
from tqdm import tqdm
import torch
from configs.deepsea_config import PARAMETERS as DEEPSEA_PARAMETERS, DeepSeaConfig
from configs.cartpole_config import PARAMETERS as CARTPOLE_PARAMETERS
from configs.config import EnvSimulationConfig
from envs.cartpole_swingup import CartpoleSwingup, CartpoleSwingupConfig
from utils.random import set_seed, create_generators, Generators
import torch.multiprocessing as mp


agents: Dict[str,
    Callable[[NDArray[np.float32], int, int, any, torch.device, Generators], Agent]] = {
    DBMRBPI.NAME: DBMRBPI.make_default_agent,
    RNDAgent.NAME: RNDAgent.make_default_agent,
    DisagreementAgent.NAME: DisagreementAgent.make_default_agent,
    APTAgent.NAME: APTAgent.make_default_agent
}

Env = Union[DeepSea, CartpoleSwingup]


def get_env(parameter: EnvSimulationConfig, generators: Generators) -> Tuple[Callable[[], Env], str]:
    """ Create results folders and callable to create the environment """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    env_cfg = parameter.env_config
    if isinstance(env_cfg, DeepSeaConfig):
        data_path = f'{dir_path}/data/{DeepSea.__name__}/{env_cfg.size}_{parameter.collection_steps}'
        make_env = lambda: DeepSea(env_cfg, generators=generators)
    elif isinstance(env_cfg, CartpoleSwingupConfig):
        data_path = f'{dir_path}/data/{CartpoleSwingup.__name__}/{parameter.collection_steps}'
        make_env = lambda: CartpoleSwingup(env_cfg, generators=generators)
    else:
        raise Exception(f'Environment config {env_cfg} not found')
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return make_env, data_path



def run(agent_name: str, agent_config: any, parameter: EnvSimulationConfig, seed: int, verbose: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    generators = create_generators(seed, device)

    env_cfg = parameter.env_config
    make_env, data_path = get_env(parameter, generators)
    env = make_env()
    s = env.reset()
    agent: Agent = agents[agent_name](len(s), env.num_actions, env.num_base_rewards, agent_config, device, generators)
    total_steps = 0
    total_upright_time = 0
    episode = 0
    tqdm_bar = tqdm(total = parameter.collection_steps)
    total_rewards = np.zeros(env.num_rewards)

    print(f'Env config {env.config}')

    while total_steps < parameter.collection_steps:
        done = False
        episode_steps = 0
        episode_rewards = np.zeros(env.num_rewards)
        episode_upright_time = 0
        while not done:
            action = agent.select_action(s, total_steps)
            timestep, info = env.step(action)
            agent.update(timestep)
            total_steps += 1
            episode_steps += 1

            
            total_rewards += timestep.rewards
            episode_rewards += timestep.rewards
            s, done = timestep.next_observation, timestep.done

            if isinstance(env, CartpoleSwingup):
                episode_upright_time += float(info['upright'])
                total_upright_time += float(info['upright'])

            if parameter.model_checkpoint_frequency is not None:
                if total_steps % parameter.model_checkpoint_frequency == 0:
                    agent.save_model(data_path, seed, total_steps)

            if total_steps >= parameter.collection_steps:
                break

        
        avg_tot_rew = total_rewards/(1+episode)
        avg_steps = total_steps/(1+episode)
        tqdm_bar.update(episode_steps)
        if isinstance(env, CartpoleSwingup):
            tqdm_bar.set_description(f'[{episode}] {episode_upright_time}/{total_upright_time} - {episode_rewards.mean()}/{episode_steps} - {avg_tot_rew.mean()}/{avg_steps}')
        else:
            tqdm_bar.set_description(f'[{episode}] {episode_rewards.mean()} - {avg_tot_rew.mean()}')
        s = env.reset()
        episode += 1
    tqdm_bar.close()
    agent.dump_buffer(data_path, env.config, seed)
    agent.save_model(data_path, seed, parameter.collection_steps)

    return


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    mp.set_start_method('spawn')
    # Change this value
    NUM_CPU = 10
    with mp.Pool(NUM_CPU) as pool:
        for parameter in CARTPOLE_PARAMETERS:
            for agent_name, agent_config in parameter.agents_config.items():
                pool.starmap(run, [(agent_name, agent_config, parameter, seed, False) for seed in parameter.seeds])
                #run(agent_name, agent_config, parameter, parameter.seeds[0], verbose=True)
    
        for parameter in DEEPSEA_PARAMETERS:
            for agent_name, agent_config in parameter.agents_config.items():
                pool.starmap(run, [(agent_name, agent_config, parameter, seed, False) for seed in parameter.seeds])
                #run(agent_name, agent_config, parameter, parameter.seeds[0], verbose=True)
    