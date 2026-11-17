"""Queue a matched temporal supervision study and its Ref-YT-VOS evaluation."""

import fcntl
import gc
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parents[1]
STUDY = PROJECT / "qfa_objective_study_20261113"
RECOVERY = PROJECT / "qfa_prefix_recovery_20261113"
REFYT = PROJECT / "qfa_target_transfer_refyt_20261113"
PROTOCOL = HERE / "protocol.json"
ARM = "mixed_prefix_temporal"
CONTROL = "mixed_prefix"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
PARENT_DIGEST = "32b7b639e3249aa4871bb0dab42b838397f37aaf917ae89f9c972346c1c9754c"
sys.path[:0] = [str(STUDY), str(REFYT), str(ROOT)]
import run_objective_study as objective
import transfer_refyt as refyt
from register import read, sha, write


def build_schedule(schedule):
    visits = Counter()
    starts = []
    for key in schedule:
        starts.append((5, 12)[visits[key] % 2])
        visits[key] += 1
    assert len(schedule) == 1536 and len(visits) == 256 and set(visits.values()) == {6}
    return starts


def preflight():
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    assert sha(STUDY / "protocol.json") == PARENT_DIGEST
    parent = objective.validate()
    schedule = parent["schedules"][CONTROL]
    starts = build_schedule(schedule)
    assert parent["future_start"] == 12 and parent["window"] == 16
    evaluation = read(REFYT / "protocol.json")
    model = Path(parent["model"])
    model_shards = {str(model / name): parent["model_sha256"][name] for name in SHARDS}
    assert all(evaluation["source_sha256"][name] == h for name, h in model_shards.items())
    assert all(sha(Path(path)) == h for path, h in evaluation["source_sha256"].items()
               if path not in model_shards)
    ds, meta = refyt.load()["source"]()
    assert len(meta) == 202 and len(evaluation["items"]) == 834 and ds["items"] == evaluation["items"]
    assert {row["video_id"] for row in parent["external_additions"]
            if row["source"] == "refytvos"}.isdisjoint(meta)
    assert sum(len(meta[row["video_id"]]["frames"]) for row in evaluation["items"]) == 22046
    eval_data = [REFYT / name for name in
                 ("official_valid.zip", "P.expression_layout.zip", "Q.expression_layout.zip", "J.expression_layout.zip")]
    previous_scores = read(REFYT / "candidate_scores.json")
    assert all(sha(path) == previous_scores["hashes"][path.name] for path in eval_data)
    paths = [Path(__file__), STUDY / "protocol.json", REFYT / "protocol.json"]
    write(PROTOCOL, {
        "scope": "Matched QFA training with alternating early and late suffix supervision",
        "parent_sha256": PARENT_DIGEST, "schedules": {ARM: schedule}, "supervision_starts": starts,
        "updates": 1536, "prefix_weight": parent["prefix_weight"],
        "seed": parent["seed"], "forward_frames": 16, "training_videos": 256,
        "source_sha256": {**evaluation["source_sha256"], **{str(path): sha(path) for path in paths}},
        "model_shards": model_shards,
        "evaluation_archive_sha256": {str(path): previous_scores["hashes"][path.name] for path in eval_data},
        "evaluation": {"dataset": "Ref-YT-VOS organizer validation", "expressions": 834,
                       "videos": 202, "mask_files_per_arm": 22046, "arms": [CONTROL, ARM],
                       "checkpoint_selection": "fixed update 1536", "windows": ["Full", "Prefix5", "Future", "Late"]},
        "gpu": {**parent["gpu"], "minimum_free_MiB": 22528},
    })
    write(HERE / "preflight.json", {"passed": True, "protocol_sha256": sha(PROTOCOL),
                                   "starts": dict(Counter(starts))})


def validate(check_model=True):
    p = read(PROTOCOL)
    assert all(sha(Path(path)) == h for path, h in p["source_sha256"].items()
               if check_model or path not in p["model_shards"])
    assert p["supervision_starts"] == build_schedule(p["schedules"][ARM])
    return p


def wait_for_permit():
    identity = {"pid": os.getpid(), "uid": os.getuid(),
                "start_ticks": Path("/proc/self/stat").read_text().split()[21],
                "uuid": os.environ["CUDA_VISIBLE_DEVICES"], "protocol_sha256": sha(PROTOCOL),
                "worker_sha256": sha(Path(__file__))}
    write(HERE / "ready.json", identity)
    while not (HERE / "permit.json").exists():
        time.sleep(.5)
    assert read(HERE / "permit.json") == identity


def verify_training(out, p):
    steps = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    assert [r["key"] for r in steps] == p["schedules"][ARM]
    assert [r["seed"] for r in steps] == list(range(p["seed"] + 1, p["seed"] + p["updates"] + 1))
    assert [r["supervised_frames"] for r in steps] == [[first + 1, first + 4] for first in p["supervision_starts"]]
    counts = Counter((r["key"], tuple(r["supervised_frames"])) for r in steps)
    assert len(counts) == 512 and set(counts.values()) == {3}
    receipt = read(out / "complete.json")
    assert receipt["updates"] == p["updates"]
    assert sha(out / "fixed/gate_best.pt") == receipt["fixed_checkpoint_sha256"]
    return receipt


def control_ready():
    directory = STUDY / f"qfa_{CONTROL}"
    return (directory / "complete.json").is_file() and (directory / "fixed/gate_best.pt").is_file()


def evaluate(arms):
    s = refyt.load()
    c = s["load_core"]()
    frozen = read(HERE / "predictions_complete.json") if (HERE / "predictions_complete.json").exists() else {}
    for arm in arms:
        if arm in frozen:
            continue
        folder = (STUDY if arm == CONTROL else HERE) / f"qfa_{arm}"
        ckpt = folder / "fixed/gate_best.pt"
        receipt = read(folder / "complete.json")
        assert receipt["updates"] == 1536 and sha(ckpt) == receipt["fixed_checkpoint_sha256"]
        out = HERE / arm
        out.mkdir(exist_ok=True)
        c["OUT"], c["QCKPT"] = out, ckpt
        receipt = read(refyt.BASELINE / "protocol.json")
        write(out / "protocol.json", {**receipt, "Q_checkpoint_sha256": sha(ckpt),
                                      "parent_protocol_sha256": sha(PROTOCOL), "label": arm})
        write(HERE / "progress.json", {"state": "refyt_inference", "arm": arm})
        c["infer"]()
        s["OUT"] = out
        s["export"]()
        frozen[arm] = read(out / "export_receipt.json")
        assert frozen[arm]["entries"] == 22046 and frozen[arm]["expressions"] == 834
        write(HERE / "predictions_complete.json", frozen)
        gc.collect()


def infer():
    import torch
    from retained_cache import RetainedCache, finish_process

    p = validate()
    cache = p["gpu"]["reusable_cache_MiB"] * (1 << 20)
    assert control_ready() and (HERE / "training_complete.json").is_file()
    refyt.load()["load_core"]()
    assert not torch.cuda.is_initialized()
    wait_for_permit()
    with RetainedCache(torch, HERE / "cache_receipt.json", target_bytes=cache, minimum_bytes=cache):
        evaluate([ARM, CONTROL])
    finish_process(0)


def score():
    import numpy as np
    import score_archives as scorer

    p = validate()
    assert all(sha(Path(path)) == h for path, h in p["evaluation_archive_sha256"].items())
    frozen = read(HERE / "predictions_complete.json")
    assert set(frozen) == {CONTROL, ARM}
    scorer.paths = lambda: [Path(path) for path in p["evaluation_archive_sha256"]]
    scorer.HERE = HERE
    scorer.main(extra_arms={arm: (Path(row["archive"]), row["sha256"]) for arm, row in frozen.items()})
    report = read(HERE / "candidate_scores.json")
    rng = np.random.default_rng(20261113)
    effects = {}
    for window, means in report["means"].items():
        paired = defaultdict(list)
        for row in report["rows"]:
            a, b = (row["scores"][arm][window] for arm in (CONTROL, ARM))
            if a is not None:
                paired[row["video_id"]].append(b["JF"] - a["JF"])
        by_video = np.array([np.mean(values) for values in paired.values()])
        draws = rng.choice(by_video, size=(10000, len(by_video)), replace=True).mean(axis=1)
        effects[window] = {"control": means[CONTROL], "temporal": means[ARM],
                           "expression_mean_JF_delta": means[ARM]["JF"] - means[CONTROL]["JF"],
                           "equal_video_JF_delta": float(by_video.mean()),
                           "video_ci95": np.quantile(draws, [.025, .975]).tolist()}
    write(HERE / "temporal_scores.json", {"dataset": report["scope"], "effects": effects,
                                          "scored_videos": report["videos"], "expressions": report["expressions"],
                                          "input_result_sha256": sha(HERE / "candidate_scores.json")})


def run():
    import evaluation_controller as ctl

    ctl.HERE, ctl.PROTOCOL, ctl.WORKER = HERE, PROTOCOL, Path(__file__)
    ctl.STATE = HERE / "evaluation_state.json"
    with (HERE / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not ctl.STATE.exists()
        assert read(HERE / "preflight.json")["protocol_sha256"] == sha(PROTOCOL)
        p = validate(check_model=False)
        try:
            ctl.state("waiting_for_checkpoints")
            while not (HERE / "training_complete.json").exists() or not control_ready():
                if not control_ready() and read(RECOVERY / "evaluation_state.json")["state"] == "failed":
                    raise RuntimeError("Matched control failed before writing its final checkpoint")
                if (HERE / "relay_failed.json").exists():
                    raise RuntimeError(read(HERE / "relay_failed.json")["reason"])
                time.sleep(30)
            ctl.state("waiting_for_model_shards")
            while not all(Path(path).is_file() for path in p["model_shards"]):
                time.sleep(30)
            validate()
            assert verify_training(HERE / f"qfa_{ARM}", p) == read(HERE / "training_complete.json")["checkpoint"]
            ctl.admit(p)
            ctl.state("scoring")
            with (HERE / "score.log").open("a") as log:
                subprocess.run([sys.executable, "-u", str(Path(__file__)), "score"], cwd=ROOT,
                               env=ctl.worker_env(""), stdout=log, stderr=subprocess.STDOUT, check=True)
            ctl.state("complete", result_sha256=sha(HERE / "temporal_scores.json"))
        except BaseException as exc:
            ctl.state("failed", reason=str(exc))
            raise


if __name__ == "__main__":
    {"preflight": preflight, "infer": infer, "score": score, "run": run}[sys.argv[1]]()
