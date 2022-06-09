import gym
import hydra
from omegaconf import DictConfig

from offline_baselines_jax import metla
from jax.config import config

import d4rl_maze

def train(cfg: DictConfig) -> None:
    env = gym.make(cfg.env_name)
    expert_data_path = cfg.expert_data_path + cfg.env_name

    if cfg.finetune:
        model = metla.METLAMSE.load(path=cfg.model_path+"-0-99", env=env)
        rewards, _ = metla.evaluate_metla(seed=cfg.seed, env=env, model=model, n_eval_episodes=10)
        model.online_finetune_setup(
            finetune=cfg.finetune_type,
            initial_rewards=rewards,
            warmup=cfg.warmup
        )
        model.learn(9999999999, log_interval=1)

    if "maze" in cfg.env_name:
        maze_batch = 1
        cfg.model.policy_kwargs.latent_dim = 2
        cfg.model.policy_kwargs.net_arch = [128] * 6
        cfg.n_batch = 50
        cfg.n_iter = 400
        maze_data_path = cfg.expert_data_path + "Maze2d_img_%d"
        expert_data_path = maze_data_path % maze_batch

    model = hydra.utils.instantiate(cfg.model, env=env)
    print("Model make")
    exit()
    model.load_data(expert_data_path)

    for batch in range(cfg.n_batch):
        for i in range(cfg.n_iter):
            model.learn(5000, reset_num_timesteps=False)
            rewards, _ = metla.evaluate_metla(
                seed=cfg.seed,
                env=env,
                model=model,
                n_eval_episodes=cfg.n_eval_episodes,
                mse="MSE" in cfg.modeltype.name
            )
            normalized_rewards = env.get_normalized_score(rewards)
            model.logger.record("eval/rewards", rewards)
            model.logger.record("eval/normalized_rewards", normalized_rewards * 100)
            model.logger.record("config/n_batch", batch, exclude="tensorboard")
            model._dump_logs()

            print("Model save:", cfg.model_path + f"-{batch}-{i}")
            model.save(cfg.model_path + f"-{batch}-{i}")

        if "maze" in cfg.env_name:
            maze_batch += 1
            if maze_batch == 18:
                maze_batch = 1
            expert_data_path = maze_data_path % maze_batch
            print("Load new data:", expert_data_path)
            model.load_data(expert_data_path)


@hydra.main(config_path="offline_baselines_jax/metla/conf", config_name="metla_conf")
def main(config: DictConfig) -> None:
    train(config)

if __name__ == "__main__":
    main()
