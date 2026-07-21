"""Diagnostic B (2026-07-21): re-gate the 2026-07-15 PPO promotion at n=2400.

The only PPO search-gate PASS ever recorded (-0.712, n=600, p=0.016)
predates the re-powered gate. This runs the pre-PPO parent milestone
(1784120250) as the CANDIDATE against the current baseline trace
(hearts_ai_search.pt == post-PPO milestone 1784156801, SDPA re-export).

Interpretation of the paired delta (candidate - baseline, negative =
candidate better): the pre-PPO net measuring +0.7 WORSE reproduces the
promotion's claimed effect; ~0 reclassifies it as a weak-gate artifact.
"""
import hashlib
import orchestrator

PRE_PPO = 'Hall_of_Fame/hearts_model_milestone_1784120250.pth'
POST_PPO = 'Hall_of_Fame/hearts_model_milestone_1784156801.pth'
BASELINE = 'hearts_model_final.pth'


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    # The gate's baseline side is hearts_ai_search.pt, the trace of the
    # CURRENT baseline - assert that really is the post-PPO milestone.
    assert _md5(BASELINE) == _md5(POST_PPO), (
        "hearts_model_final.pth is not the post-PPO milestone; "
        "the re-gate would measure the wrong pair")
    print("Baseline identity verified: hearts_model_final.pth == milestone 1784156801")
    success, mean, p = orchestrator.evaluate_candidate_search(
        PRE_PPO, deals=2400, k=32, alpha=0.05)
    print(f"RE-GATE RESULT: pre-PPO-vs-post-PPO delta {mean:+.3f}, p={p:.5f} "
        f"(positive delta = post-PPO baseline genuinely stronger)")


if __name__ == '__main__':
    main()
