"""Equivalence + performance harness for hearts_env.HeartsVecEnv.

Equivalence: N vec envs vs N single envs with identical seeds, stepped with
identical scripted-random legal actions through full games - observations,
masks, labels, and terminal scores must be bit-identical at every decision
(both paths wrap the same HeartsEnv with the same per-env RNG seeding).

Benchmark: per-decision cost of the single-env list-marshaling API vs the
batched numpy API - the number that justified building this.

Run:  python test_vecenv.py
"""

import random
import sys
import time

import numpy as np

sys.path.insert(0, 'build/Release')
import hearts_env


def random_legal(mask_row, rng):
    legal = np.flatnonzero(mask_row)
    return int(legal[rng.randrange(len(legal))])


def test_equivalence(n_envs=8, games_per_env=25, seed0=7700):
    vec = hearts_env.HeartsVecEnv(n_envs, seed0)
    singles = [hearts_env.HeartsEnv(seed=seed0 + i) for i in range(n_envs)]
    for e in singles:
        e.reset()

    rngs = [random.Random(1234 + i) for i in range(n_envs)]
    games_done = [0] * n_envs
    all_idx = np.arange(n_envs, dtype=np.int64)
    decisions = 0

    while min(games_done) < games_per_env:
        cp = vec.current_players()
        obs = vec.observe_batch(all_idx)
        mask = vec.legal_mask_batch(all_idx)
        labels = vec.labels_batch(all_idx)

        actions = np.empty(n_envs, dtype=np.int64)
        for i in range(n_envs):
            # single-env reference for env i
            assert singles[i].get_current_player() == int(cp[i]), "current_player diverged"
            s_obs = np.asarray(singles[i].observe(), dtype=np.float32)
            assert np.array_equal(s_obs, obs[i]), f"obs diverged at env {i}"
            s_mask = np.zeros(52, dtype=bool)
            for a in singles[i].get_legal_actions():
                if a != -1:
                    s_mask[a] = True
            assert np.array_equal(s_mask, mask[i]), f"mask diverged at env {i}"
            s_lab = np.asarray(singles[i].observe_opponent_hands(), dtype=np.float32)
            assert np.array_equal(s_lab, labels[i]), f"labels diverged at env {i}"
            actions[i] = random_legal(mask[i], rngs[i])
            decisions += 1

        dones, scores = vec.step_batch(all_idx, actions)
        for i in range(n_envs):
            r = singles[i].step(int(actions[i]))
            assert bool(dones[i]) == r.done, f"done flag diverged at env {i}"
            if r.done:
                s_scores = singles[i].get_round_scores()
                assert list(scores[i]) == list(s_scores), f"scores diverged at env {i}"
                singles[i].reset()  # mirror the vec auto-reset
                games_done[i] += 1

    print(f"EQUIVALENCE PASS: {sum(games_done)} games, {decisions} decisions, "
          f"obs/mask/labels/scores bit-identical")


def benchmark(n_envs=128, rounds=2000, seed0=31000):
    vec = hearts_env.HeartsVecEnv(n_envs, seed0)
    singles = [hearts_env.HeartsEnv(seed=seed0 + i) for i in range(n_envs)]
    for e in singles:
        e.reset()
    all_idx = np.arange(n_envs, dtype=np.int64)

    t0 = time.perf_counter()
    for _ in range(rounds):
        obs = vec.observe_batch(all_idx)
        mask = vec.legal_mask_batch(all_idx)
    vec_us = (time.perf_counter() - t0) / (rounds * n_envs) * 1e6

    t0 = time.perf_counter()
    for _ in range(rounds // 10):  # the slow path needs fewer reps
        for e in singles:
            o = np.asarray(e.observe(), dtype=np.float32)
            m = np.zeros(52, dtype=bool)
            for a in e.get_legal_actions():
                if a != -1:
                    m[a] = True
    single_us = (time.perf_counter() - t0) / ((rounds // 10) * n_envs) * 1e6

    print(f"BENCHMARK (obs+mask per decision): single-env list API "
          f"{single_us:.1f} us | batched numpy {vec_us:.2f} us | "
          f"{single_us / vec_us:.0f}x")


if __name__ == '__main__':
    test_equivalence()
    benchmark()
