import gymnasium as gym

def main():
    env = gym.make("Hopper-v4")
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    print("SMOKE_OK")

if __name__ == "__main__":
    main()