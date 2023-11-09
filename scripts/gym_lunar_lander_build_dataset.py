from typing import List
import argparse
import random
import json
import os

from torch.distributions.categorical import Categorical
from PIL import Image
from datasets import Dataset
import gymnasium as gym
import torch.nn as nn
import numpy as np
import torch

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER

LUNAR_LANDER_OPTIONS = (
    "[FIRE LEFT ENGINE], [FIRE RIGHT ENGINE], [FIRE MAIN ENGINE], [NOTHING]".split(", ")
)

MAX_STEPS = 1000


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def _gen_examples(round_num, args):
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    random.seed(round_num)
    np.random.seed(round_num)

    class EnvWrapper:
        single_observation_space = env.observation_space
        single_action_space = env.action_space

    model = Agent(EnvWrapper()).to("cpu")
    model.load_state_dict(
        torch.load(args.pretrained_ppo_model_path, map_location="cpu")
    )
    model.eval()

    os.makedirs(args.output_image_folder, exist_ok=True)

    observation, info = env.reset(seed=round_num)

    for frame in range(MAX_STEPS):
        img = env.render()
        with torch.no_grad():
            action, logprob, _, value = model.get_action_and_value(
                torch.from_numpy(observation)
            )

        action = action.cpu().numpy()
        resp = ""
        if action == 0:
            resp = "[NOTHING]"
        elif action == 1:
            resp = "[FIRE LEFT ENGINE]"
        elif action == 2:
            resp = "[FIRE MAIN ENGINE]"
        elif action == 3:
            resp = "[FIRE RIGHT ENGINE]"
        if random.random() < args.sample_rate:
            random.shuffle(LUNAR_LANDER_OPTIONS)
            options_str = ", ".join(LUNAR_LANDER_OPTIONS)
            img_fn = os.path.join(args.output_image_folder, f"{round_num}_{frame}.jpg")
            messages = [
                {
                    "role": ROLE_USER,
                    "content": f"<image>\nYou are playing lunar lander. The goal is to land the craft between the yellow flags. What is the optimal next action? {options_str}",
                },
                {"role": ROLE_ASSISTANT, "content": resp},
            ]
            Image.fromarray(img).save(img_fn)
            example = {
                "id": f"{round_num}_{frame}",
                "images": [img_fn],
                "messages": messages,
            }
            yield example

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break


def main(args):
    def gen(idxs):
        for r in idxs:
            yield from _gen_examples(r, args)

    ds = Dataset.from_generator(
        gen, gen_kwargs={"idxs": list(range(args.rounds))}, num_proc=args.num_proc
    )
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ppo_model_path", type=str)
    parser.add_argument("--output_image_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--rounds", type=int, default=10_000)
    parser.add_argument("--sample_rate", type=float, default=0.01)
    parser.add_argument("--num_proc", type=int, default=16)
    args = parser.parse_args()
    main(args)
