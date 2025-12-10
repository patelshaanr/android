import gymnasium as gym
import torch
from torch.distributions import Categorical

from model import A3CNet


def worker_process(global_net, optimizer, global_counter, worker_id, args, log_queue):
    print(f"[WORKER {worker_id}] Starting worker process.", flush=True)

    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n

    local_net = A3CNet(num_actions)
    local_net.load_state_dict(global_net.state_dict())

    gamma = args["gamma"]
    t_max = args["t_max"]
    entropy_beta = args["entropy_beta"]
    value_loss_coef = args["value_loss_coef"]
    grad_clip = args["grad_clip"]

    episode_idx = 0

    while True:
        with global_counter.get_lock():
            if global_counter.value >= args["max_steps"]:
                print(f"[WORKER {worker_id}] Reached max_steps, exiting loop.", flush=True)
                break

        local_net.load_state_dict(global_net.state_dict())

        log_probs = []
        values = []
        rewards = []
        entropies = []

        state, _ = env.reset()
        done = False
        t = 0
        ep_return = 0.0
        episode_idx += 1

        while t < t_max and not done:
            s = torch.from_numpy(state).unsqueeze(0)  # [1, 4]

            logits, value = local_net(s)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(float(reward))
            entropies.append(entropy)

            ep_return += float(reward)
            state = next_state
            t += 1

            with global_counter.get_lock():
                global_counter.value += 1
                steps_now = global_counter.value
                if steps_now % 10_000 == 0:
                    print(f"[WORKER {worker_id}] global_steps={steps_now}", flush=True)
                if global_counter.value >= args["max_steps"]:
                    done = True
                    break

        if done:
            R = 0.0
        else:
            with torch.no_grad():
                s = torch.from_numpy(state).unsqueeze(0)
                _, v = local_net(s)
                R = float(v.item())

        policy_loss = 0.0
        value_loss = 0.0

        for r, v, log_p, ent in reversed(list(zip(rewards, values, log_probs, entropies))):
            R = r + gamma * R
            v = v.squeeze()
            target = torch.tensor(R, dtype=torch.float32)
            advantage = target - v

            value_loss = value_loss + 0.5 * (advantage.pow(2))
            policy_loss = policy_loss - (log_p * advantage.detach() + entropy_beta * ent)

        total_loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad()
        total_loss.backward()

        for param in global_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

        optimizer.step()

        log_queue.put(ep_return)

        if episode_idx % 10 == 0:
            print(f"[WORKER {worker_id}] Episode {episode_idx}, return={ep_return}", flush=True)

    env.close()
    print(f"[WORKER {worker_id}] Closed env and exiting.", flush=True)