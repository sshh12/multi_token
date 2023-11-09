import argparse
import random
import requests
import os

from PIL import Image
import gymnasium as gym

from multi_token.constants import ROLE_USER

LUNAR_LANDER_OPTIONS = (
    "[FIRE LEFT ENGINE], [FIRE RIGHT ENGINE], [FIRE MAIN ENGINE], [NOTHING]".split(", ")
)

MAX_STEPS = 1000


def main(args):
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, args.video_folder)
    env.reset()

    for _ in range(MAX_STEPS):
        img = env.render()
        random.shuffle(LUNAR_LANDER_OPTIONS)
        options_str = ", ".join(LUNAR_LANDER_OPTIONS)
        img_fn = os.path.join("/tmp", "frame.jpg")
        messages = [
            {
                "role": ROLE_USER,
                "content": f"<image>\nYou are playing lunar lander. The goal is to land the craft between the yellow flags. What is the optimal next action? {options_str}",
            },
        ]
        Image.fromarray(img).save(img_fn)
        example = {
            "images": [img_fn],
            "messages": messages,
        }
        output = requests.post(
            args.server_endpoint,
            json=example,
        ).json()["output"]
        print("> " + output)
        if output == "[FIRE LEFT ENGINE]":
            action = 1
        elif output == "[FIRE MAIN ENGINE]":
            action = 2
        elif output == "[FIRE RIGHT ENGINE]":
            action = 3
        else:
            action = 0

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_endpoint", type=str, default="http://localhost:7860/generate"
    )
    parser.add_argument(
        "--video_folder", type=str, default="/data/gym_lunar_lander_video"
    )
    args = parser.parse_args()
    main(args)
