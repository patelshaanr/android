import time
import torch
import torch.multiprocessing as mp

from model import A3CNet
from worker import worker_process


def run_multiprocess():
    print("[MAIN] Starting A3C CartPole training (multiprocess)...", flush=True)

    num_workers = 4
    num_actions = 2  


    global_net = A3CNet(num_actions)
    global_net.share_memory()
    print("[MAIN] Global network created and shared.", flush=True)

    optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-4)
    print("[MAIN] Optimizer created.", flush=True)

    args = {
        "gamma": 0.99,
        "t_max": 20,
        "entropy_beta": 0.01,
        "value_loss_coef": 0.5,
        "grad_clip": 40.0,
        "max_steps": 200_000,
    }

    global_counter = mp.Value("i", 0)
    log_queue = mp.Queue()

    workers = []
    for wid in range(num_workers):
        print(f"[MAIN] Starting worker {wid}...", flush=True)
        p = mp.Process(
            target=worker_process,
            args=(global_net, optimizer, global_counter, wid, args, log_queue),
        )
        p.start()
        workers.append(p)

    print("[MAIN] All workers started. Entering logging loop.", flush=True)

    episode_returns = []
    last_print_time = time.time()

    while any(p.is_alive() for p in workers):
        while not log_queue.empty():
            r = log_queue.get()
            episode_returns.append(r)
            print(f"[MAIN] [Episode {len(episode_returns)}] Return = {r}", flush=True)

        now = time.time()
        if now - last_print_time > 5:
            with global_counter.get_lock():
                steps = global_counter.value
            print(
                f"[MAIN] Still training... global_steps={steps}, episodes={len(episode_returns)}",
                flush=True,
            )
            last_print_time = now

        time.sleep(0.5)

    print("[MAIN] All workers finished. Joining...", flush=True)
    for p in workers:
        p.join()

    print(f"[MAIN] Training complete. Total episodes: {len(episode_returns)}", flush=True)

    with open("training_returns.txt", "w") as f:
        for r in episode_returns:
            f.write(f"{r}\n")

    print("[MAIN] Saved episode returns to training_returns.txt", flush=True)


def run_single_worker_debug():
    """Run a single worker in the main process to debug logic without multiprocessing."""
    print("[MAIN-DEBUG] Running single-worker debug mode...", flush=True)

    num_actions =2
    global_net = A3CNet(num_actions)

    optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-4)
    args = {
        "gamma": 0.99,
        "t_max": 20,
        "entropy_beta": 0.01,
        "value_loss_coef": 0.5,
        "grad_clip": 40.0,
        "max_steps": 5_000,
    }

    global_counter = mp.Value("i", 0)
    log_queue = mp.Queue()

    from worker import worker_process
    worker_process(global_net, optimizer, global_counter, worker_id=0, args=args, log_queue=log_queue)

    print("[MAIN-DEBUG] Worker finished. global_steps =", global_counter.value, flush=True)
    while not log_queue.empty():
        r = log_queue.get()
        print("[MAIN-DEBUG] Episode return:", r, flush=True)


if __name__ == "__main__":
 
    mp.set_start_method("spawn", force=True)


    run_multiprocess()

