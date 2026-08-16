"""Fast LIBERO oracle test for best-of-K action-chunk candidates."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from lerobot_policy_bitwam.bitvla_evaluate import _load_upstream_evaluator
from lerobot_policy_bitwam.workflows import load_config


def make_action_candidates(
    actions: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    action_std: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    noise_scale: float,
) -> np.ndarray:
    """Return nested candidates with the untouched policy action at index zero."""
    if count < 1:
        raise ValueError("candidate_count must be positive")
    base = np.asarray(actions, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != 7:
        raise ValueError(f"Expected an [H, 7] action chunk, got {base.shape}")

    candidates = np.repeat(base[None], count, axis=0)
    if count == 1 or noise_scale == 0:
        return candidates

    noise = rng.normal(size=(count - 1, *base.shape)).astype(np.float32)
    # Adjacent actions form a smooth perturbation rather than eight unrelated jolts.
    padded = np.pad(noise[:, :, :6], ((0, 0), (1, 1), (0, 0)), mode="edge")
    noise[:, :, :6] = (
        0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    )
    noise[:, :, 6] = 0.0  # Gripper timing is discrete; preserve the trained policy's choice.
    candidates[1:] += noise * np.asarray(action_std, dtype=np.float32) * noise_scale
    candidates[1:] = np.clip(candidates[1:], action_low, action_high)
    candidates[1:, :, 6] = base[:, 6]
    return candidates


def goal_progress(env: Any) -> tuple[int, int]:
    """Count satisfied LIBERO goal predicates using the benchmark's own evaluator."""
    raw_env = env.env
    goals = raw_env.parsed_problem["goal_state"]
    return sum(bool(raw_env._eval_predicate(goal)) for goal in goals), len(goals)


def _restore(env: Any, state: np.ndarray, timestep: int) -> Any:
    obs = env.regenerate_obs_from_state(state)
    env.env.timestep = timestep
    env.env.done = False
    return obs


def _evaluate_candidate(env: Any, evaluator: Any, cfg: Any, actions: np.ndarray) -> tuple[int, bool]:
    success = False
    for action in actions:
        processed = evaluator.process_action(action, cfg.model_family)
        _, _, done, _ = env.step(processed.tolist())
        if done:
            success = True
            break
    progress, total = goal_progress(env)
    if success:
        progress = total
    return progress, success


def _branch_worker(config: dict[str, Any], task_id: int, connection: Connection) -> None:
    """Own the branch simulator in another process so it cannot mutate control globals."""
    env = None
    try:
        evaluator = _load_upstream_evaluator(config)
        cfg = SimpleNamespace(model_family="bitnet")
        task_suite_name = str(config.get("task_suite_name", "libero_10"))
        task_suite = evaluator.benchmark.get_benchmark_dict()[task_suite_name]()
        task = task_suite.get_task(task_id)
        env = evaluator.get_libero_env(task, cfg.model_family, resolution=256)[0]
        connection.send({"ready": True})
        while True:
            payload = connection.recv()
            if payload is None:
                break
            state, timestep, candidates = payload
            scores = []
            for candidate in candidates:
                _restore(env, state, timestep)
                score, _ = _evaluate_candidate(env, evaluator, cfg, candidate)
                scores.append(score)
            connection.send(scores)
    except BaseException as exc:
        with suppress(BrokenPipeError, EOFError):
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if env is not None:
            env.close()
        connection.close()


def _start_branch_worker(config: dict[str, Any], task_id: int) -> tuple[mp.Process, Connection]:
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_branch_worker,
        args=(config, task_id, child_connection),
        daemon=True,
    )
    process.start()
    child_connection.close()
    if not parent_connection.poll(180):
        process.terminate()
        process.join(10)
        raise RuntimeError("Timed out starting isolated LIBERO branch simulator")
    ready = parent_connection.recv()
    if ready != {"ready": True}:
        process.join(10)
        raise RuntimeError(f"Isolated LIBERO branch simulator failed: {ready}")
    return process, parent_connection


def run_episode(
    *,
    evaluator: Any,
    cfg: Any,
    env: Any,
    branch_connection: Connection | None,
    task_description: str,
    model: Any,
    resize_size: Any,
    processor: Any,
    action_head: Any,
    proprio_projector: Any,
    noisy_action_projector: Any,
    initial_state: np.ndarray,
    candidate_count: int,
    noise_scale: float,
    action_stats: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run one paired oracle episode; ties always retain the base policy action."""
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue: deque[np.ndarray] = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = evaluator.TASK_MAX_STEPS[cfg.task_suite_name]
    action_std = np.asarray(action_stats["std"], dtype=np.float32)
    action_low = np.asarray(action_stats["q01"], dtype=np.float32)
    action_high = np.asarray(action_stats["q99"], dtype=np.float32)
    t = 0
    decisions = 0
    nonbase_wins = 0
    strict_progress_wins = 0
    branch_steps = 0
    success = False

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(evaluator.get_libero_dummy_action(cfg.model_family))
            if done:
                success = True
                break
            t += 1
            continue

        observation, _ = evaluator.prepare_observation(obs, resize_size)
        if not action_queue:
            base_actions = np.asarray(
                evaluator.get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                ),
                dtype=np.float32,
            )
            if candidate_count == 1:
                action_queue.extend(base_actions)
                decisions += 1
                continue
            candidates = make_action_candidates(
                base_actions,
                count=candidate_count,
                rng=rng,
                action_std=action_std,
                action_low=action_low,
                action_high=action_high,
                noise_scale=noise_scale,
            )
            state = env.get_sim_state().copy()
            timestep = int(env.env.timestep)
            if branch_connection is None:
                raise RuntimeError("Multiple candidates require an isolated branch environment")
            branch_connection.send((state, timestep, candidates))
            if not branch_connection.poll(180):
                raise RuntimeError("Timed out waiting for isolated branch scores")
            scores = branch_connection.recv()
            if isinstance(scores, dict) and "error" in scores:
                raise RuntimeError(f"Isolated branch simulator failed: {scores['error']}")
            branch_steps += len(candidates) * len(base_actions)
            selected = int(np.argmax(scores))
            if selected:
                nonbase_wins += 1
                if scores[selected] > scores[0]:
                    strict_progress_wins += 1
            action_queue.extend(candidates[selected])
            decisions += 1

        action = evaluator.process_action(action_queue.popleft(), cfg.model_family)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    progress, total_goals = goal_progress(env)
    return {
        "success": success,
        "steps": t,
        "goal_progress": progress,
        "goal_count": total_goals,
        "decisions": decisions,
        "nonbase_wins": nonbase_wins,
        "strict_progress_wins": strict_progress_wins,
        "branch_steps": branch_steps,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    evaluator = _load_upstream_evaluator(config)
    candidate_count = int(config.get("candidate_count", 1))
    trials = int(config.get("trials_per_task", 1))
    task_ids = [int(value) for value in config.get("task_ids", range(10))]
    seed = int(config.get("seed", 7))
    noise_scale = float(config.get("noise_scale", 0.15))

    cfg = evaluator.GenerateConfig(
        model_family="bitnet",
        pretrained_checkpoint=str(config["checkpoint"]),
        task_suite_name=str(config.get("task_suite_name", "libero_10")),
        num_trials_per_task=trials,
        use_wandb=False,
        seed=seed,
    )
    evaluator.validate_config(cfg)
    evaluator.set_seed_everywhere(seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = (
        evaluator.initialize_model(cfg)
    )
    model.set_constant(
        image_token_idx=evaluator.BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=evaluator.BITNET_PROPRIO_PAD_IDX,
        ignore_idx=evaluator.BITNET_IGNORE_INDEX,
        action_token_begin_idx=evaluator.BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=evaluator.BITNET_STOP_INDEX,
    )
    resize_size = evaluator.get_image_resize_size(cfg)
    task_suite = evaluator.benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    action_stats = model.norm_stats[cfg.unnorm_key]["action"]

    started = time.time()
    episodes: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, description = evaluator.get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        branch_process = None
        branch_connection = None
        if candidate_count > 1:
            branch_process, branch_connection = _start_branch_worker(config, task_id)
        try:
            for episode_id in range(trials):
                episode_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, task_id, episode_id])
                )
                result = run_episode(
                    evaluator=evaluator,
                    cfg=cfg,
                    env=env,
                    branch_connection=branch_connection,
                    task_description=description,
                    model=model,
                    resize_size=resize_size,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    initial_state=initial_states[episode_id],
                    candidate_count=candidate_count,
                    noise_scale=noise_scale,
                    action_stats=action_stats,
                    rng=episode_rng,
                )
                result.update({"task_id": task_id, "episode_id": episode_id, "task": description})
                episodes.append(result)
                print("ORACLE_EPISODE " + json.dumps(result, sort_keys=True), flush=True)
        finally:
            env.close()
            if branch_connection is not None:
                with suppress(BrokenPipeError, EOFError):
                    branch_connection.send(None)
                branch_connection.close()
            if branch_process is not None:
                branch_process.join(10)
                if branch_process.is_alive():
                    branch_process.terminate()
                    branch_process.join(10)

    successes = sum(int(item["success"]) for item in episodes)
    summary = {
        "checkpoint": str(config["checkpoint"]),
        "upstream_revision": str(config["upstream_revision"]),
        "candidate_count": candidate_count,
        "noise_scale": noise_scale,
        "seed": seed,
        "successes": successes,
        "episodes": len(episodes),
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "nonbase_wins": sum(int(item["nonbase_wins"]) for item in episodes),
        "strict_progress_wins": sum(int(item["strict_progress_wins"]) for item in episodes),
        "elapsed_seconds": time.time() - started,
        "episode_results": episodes,
    }
    output_path = Path(config["output_path"]).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("ORACLE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--trials-per-task", type=int)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.task_id is not None:
        config["task_ids"] = [args.task_id]
    if args.candidate_count is not None:
        config["candidate_count"] = args.candidate_count
    if args.trials_per_task is not None:
        config["trials_per_task"] = args.trials_per_task
    if args.output_path is not None:
        config["output_path"] = str(args.output_path)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(config)


if __name__ == "__main__":
    main()
