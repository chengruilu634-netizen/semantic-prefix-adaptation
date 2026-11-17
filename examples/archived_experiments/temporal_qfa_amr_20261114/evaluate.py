"""Swap only the QFA checkpoint in the frozen SPA evaluation suite."""

import ast
import fcntl
import hashlib
import inspect
import math
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

WORKER = Path(__file__).absolute()
SUITE = WORKER.resolve().parent
PROJECT = SUITE.parent
ROOT = PROJECT.parents[1]
REFYT = PROJECT / "qfa_target_transfer_refyt_20261113"
REFSAV = PROJECT / "refsav_refyt_baselines_20261113/refsav"
V4B = PROJECT / "improvement_v4b_exp059Q_M"
PAPER = PROJECT / "paper_supplement_20261006"
TEMPORAL = PROJECT / "qfa_temporal_supervision_20261113"
QFA = TEMPORAL / "qfa_mixed_prefix_temporal/fixed/gate_best.pt"
GPU = "GPU-c9fea242-043b-ec16-abb2-d59b8141063a"
WINDOWS = ("Full", "Prefix5", "Future", "Late")
LABEL = "temporal_spa"
sys.path[:0] = [str(REFYT), str(REFSAV), str(PROJECT / "qfa_mevis_scale_20261110"),
                str(PROJECT / "qfa_mevis_scale_eval_20261110"), str(ROOT)]
from register import read, sha, write
from transfer_refyt import definitions

BASELINES = {
    "mevis": V4B / "external_mevis_u_Mon_v1",
    "refyt": PAPER / "eval_refyt_J_native_r0_p5",
    "refsav": REFSAV / "eval_J",
    "davis": PAPER / "eval_davis_J_native_r0_p5",
    "revos": V4B / "external_revos_Mon_v1",
}
COUNTS = {"mevis": 793, "refyt": 834, "refsav": 1945, "davis": 244, "revos": 5822}
ORDER = ("mevis", "refyt", "refsav", "davis", "revos")


def runner(dataset, out):
    if dataset in ("mevis", "revos"):
        path = PROJECT / f"improvement_v3/eval_{'mevis_u' if dataset == 'mevis' else 'revos'}_exp059q_m_v4b.py"
        core = definitions(path, ["infer", "on"], f"spa_{dataset}")
        core["OUT"] = out
        return core, None
    if dataset == "refsav":
        import run_refsav

        core = run_refsav.load_core()
        core["OUT"] = out
        return core, run_refsav
    path = PROJECT / "supplement_eval_20261006.py"
    adapter = definitions(path, ["infer", dataset, "J", "native", "0", "5"], f"spa_{dataset}")
    adapter["OUT"] = out
    return adapter["load_core"](), adapter


def register(dataset):
    out = SUITE / dataset
    original = BASELINES[dataset]
    core, adapter = runner(dataset, out)
    parent = read(original / "protocol.json")
    frozen = read(original / "prediction_freeze.json")
    assert frozen["state"] == "complete" and frozen["GT_access"] is False
    assert frozen["protocol_sha256"] == sha(original / "protocol.json")
    assert len(frozen["files"]) == COUNTS[dataset]
    assert sha(core["QCKPT"]) == parent["Q_checkpoint_sha256"]
    assert sha(core["MCKPT"]) == parent["M_checkpoint_sha256"]
    qfa_sha = sha(QFA)
    receipt = QFA.parents[1] / "complete.json"
    assert read(receipt)["updates"] == 1536
    assert read(receipt)["fixed_checkpoint_sha256"] == qfa_sha
    ds, meta = core["source"]()
    items = core["manifest"](ds) if dataset in ("mevis", "revos") else ds["items"]
    assert items == parent["items"]
    assert sha(ROOT / ds["meta"]) == parent["metadata_sha256"]
    assert {row["key"] for row in items} == {row["key"] for row in frozen["files"]}
    if dataset in ("refyt", "refsav"):
        train = read(PROJECT / "qfa_target_mix_20261113/protocol.json")
        source = "refytvos" if dataset == "refyt" else "refsav"
        used = {r["video_id"] for r in train["external_additions"] if r["source"] == source}
        assert len(used) == 64 and used.isdisjoint(meta)
    pins = {**parent.get("code_sha256", {}), **parent.get("source_sha256", {})}
    for path, digest in pins.items():
        assert sha(Path(path)) == digest, path
    for path in (WORKER.resolve(), receipt, QFA, core["MCKPT"], Path(core["__file__"]),
                 original / "protocol.json", original / "prediction_freeze.json",
                 PROJECT / "qfa_mevis_scale_eval_20261110/evaluation_controller.py",
                 PROJECT / "qfa_mevis_scale_20261110/retained_cache.py"):
        pins[str(path)] = sha(path)
    if isinstance(adapter, dict):
        path = Path(adapter["__file__"])
        pins[str(path)] = sha(path)
    elif adapter is not None:
        path = Path(adapter.__file__)
        pins[str(path)] = sha(path)
    summary = original / "summary.json"
    if dataset == "refsav":
        summary = REFSAV / "comparison.json"
    if dataset == "refyt":
        summary = TEMPORAL / "candidate_scores.json"
        for path in (REFYT / "score_archives.py", REFYT / "davis_metrics.py"):
            pins[str(path)] = sha(path)
    pins[str(summary)] = sha(summary)
    protocol = {**parent, "items": items, "arm": "J", "GT_access": False,
                "label": LABEL, "method": "Temporal QFA with frozen original v4b AMR",
                "Q_checkpoint_sha256": qfa_sha, "original_Q_checkpoint_sha256": parent["Q_checkpoint_sha256"],
                "reference_summary": str(summary), "source_sha256": pins,
                "gpu": {"uuid": GPU, "minimum_free_MiB": 24576, "reusable_cache_MiB": 20480}}
    out.mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    entry = out / WORKER.name
    if not entry.exists():
        entry.symlink_to(WORKER.resolve())
    path = out / "protocol.json"
    if path.exists():
        assert read(path) == protocol
    else:
        write(path, protocol)
    print(dataset, len(items), sha(path), flush=True)


def validate(out):
    p = read(out / "protocol.json")
    for path, digest in p["source_sha256"].items():
        assert sha(Path(path)) == digest, path
    return p


def infer(out):
    from retained_cache import RetainedCache, finish_process

    p = validate(out)
    core, _ = runner(out.name, out)
    core["QCKPT"] = QFA
    torch = core["torch"]
    assert not torch.cuda.is_initialized()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU
    identity = {"pid": os.getpid(), "uid": os.getuid(),
                "start_ticks": Path("/proc/self/stat").read_text().split()[21],
                "uuid": GPU, "protocol_sha256": sha(out / "protocol.json"), "worker_sha256": sha(WORKER)}
    write(out / "ready.json", identity)
    while not (out / "permit.json").exists():
        time.sleep(.5)
    assert read(out / "permit.json") == identity
    cache = p["gpu"]["reusable_cache_MiB"] << 20
    with RetainedCache(torch, out / "cache_receipt.json", target_bytes=cache, minimum_bytes=cache):
        if out.name in ("refyt", "davis"):
            _, adapter = runner(out.name, out)
            adapter["load_core"] = lambda: core
            adapter["infer"]()
        else:
            core["infer"]()
        verify_freeze(out)
    finish_process(0)


def verify_freeze(out):
    p = read(out / "protocol.json")
    freeze = read(out / "prediction_freeze.json")
    assert freeze["state"] == "complete" and not freeze["GT_access"]
    assert freeze["protocol_sha256"] == sha(out / "protocol.json")
    assert [row["key"] for row in p["items"]] == [row["key"] for row in freeze["files"]]


def splice(function, substitutions, **globals_):
    code = textwrap.dedent(inspect.getsource(function))
    for original, replacement in substitutions:
        assert code.count(original) == 1, original
        code = code.replace(original, replacement)
    scope = {**function.__globals__, **globals_}
    exec(compile(ast.parse(code), function.__code__.co_filename, "exec"), scope)
    return scope[function.__name__]


def archived_score_with_windows(dataset, out):
    # Add frame-window means to the frozen MeViS/ReVOS scorers, retaining their full metric arithmetic.
    core, _ = runner(dataset, out)
    indent = "\n        "
    replacements = []
    for var, metric in (("j", "iou"), ("f", "boundary")):
        call = f"db_eval_{metric}(truth, pred)"
        old = f"{var} = float({call}.mean()) * 100"
        new = f"{var}_frames = {call}{indent}{var} = float({var}_frames.mean()) * 100"
        if dataset == "revos":
            replacements.append((old, new))
    if dataset == "mevis":
        old = "j, f = float(db_eval_iou(truth, pred).mean()) * 100, float(db_eval_boundary(truth, pred).mean()) * 100"
        replacements.append((old, indent.join((
            "j_frames = db_eval_iou(truth, pred)", "f_frames = db_eval_boundary(truth, pred)",
            "j, f = float(j_frames.mean()) * 100, float(f_frames.mean()) * 100"))))
    row_end = '"R": r})' if dataset == "revos" else '"JF": (j + f) / 2})'
    replacements.append((row_end, row_end + indent + "rows[-1]['windows'] = window_metrics(j_frames, f_frames)"))
    splice(core["score"], replacements, window_metrics=window_metrics)()
    return read(out / "summary.json")


def window_metrics(j, f):
    n = len(j)
    spans = ((0, n), (0, min(5, n)), (5, n), (max(5, math.ceil(.75*n)), n))
    windows = {}
    for label, (start, stop) in zip(WINDOWS, spans):
        if stop > start:
            jm, fm = float(j[start:stop].mean())*100, float(f[start:stop].mean())*100
            windows[label] = {"J": jm, "F": fm, "JF": (jm + fm)/2}
    return windows


def score_refsav(out):
    import run_refsav as base

    scoring = out / "scoring"
    scoring.mkdir(exist_ok=True)
    freezes = {}
    for arm, source in (("P", REFSAV / "eval_P"), ("J", BASELINES["refsav"]), (LABEL, out)):
        link = scoring / f"eval_{arm}"
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        freezes[arm] = sha(source / "prediction_freeze.json")
    write(scoring / "predictions_complete.json", freezes)
    splice(base.score, [('pairs = [("P", "Q"), ("P", "J"), ("Q", "J")]',
                         'pairs = [("P", "J"), ("J", "temporal_spa")]')],
           HERE=scoring, ARMS=("P", "J", LABEL))()
    return read(scoring / "comparison.json")


def score_refyt(out):
    import score_archives as scorer

    _, adapter = runner("refyt", out)
    if not (out / "export_receipt.json").exists():
        adapter["export"]()
    combined = read(out / "export_receipt.json")
    paths = scorer.paths()
    scorer.HERE = out
    scorer.paths = lambda: paths
    metrics = out / "davis_metrics.py"
    if not metrics.exists():
        metrics.symlink_to(REFYT / metrics.name)
    scorer.main(extra_arms={LABEL: (Path(combined["archive"]), combined["sha256"])})
    return read(out / "candidate_scores.json")


def paired(reference, candidate, by_type=False):
    import numpy as np

    assert reference.keys() == candidate.keys()
    rng = np.random.default_rng(20261114)
    result = {}
    for window in WINDOWS:
        windows = [(reference[k], candidate[k]) for k in reference
                   if (window in reference[k]["windows"] and reference[k]["windows"][window] is not None)]
        rows = [(a, b, a["windows"][window], b["windows"][window]) for a, b in windows]
        assert rows and all(a["video_id"] == b["video_id"] for a, b, _, _ in rows)
        metrics = {}
        for name, index in (("original_spa", 2), (LABEL, 3)):
            mean = lambda rr: {key: float(np.mean([row[index][key] for row in rr])) for key in ("J", "F", "JF")
                               if key in rr[0][index]}
            if by_type:
                groups = [mean([r for r in rows if r[0]["type_id"] == category]) for category in (0, 1)]
                metrics[name] = {key: (groups[0][key] + groups[1][key])/2 for key in groups[0]}
            else:
                metrics[name] = mean(rows)
        videos = {}
        for a, _, before, after in rows:
            if by_type and a["type_id"] not in (0, 1):
                continue
            videos.setdefault(a["video_id"], []).append(after["JF"] - before["JF"])
        effects = np.array([np.mean(v) for v in videos.values()])
        draws = rng.choice(effects, size=(10000, len(effects)), replace=True).mean(1)
        result[window] = {**metrics, "delta_JF": metrics[LABEL]["JF"] - metrics["original_spa"]["JF"],
                          "video_mean_delta": float(effects.mean()),
                          "video_ci95": np.quantile(draws, [.025, .975]).tolist(),
                          "improved_degraded_tied": [int((effects > 1e-6).sum()), int((effects < -1e-6).sum()),
                                                     int((abs(effects) <= 1e-6).sum())]}
    return result


def score(out):
    dataset = out.name
    p = validate(out)
    verify_freeze(out)
    baseline = BASELINES[dataset]
    original_summary = read(Path(p["reference_summary"]))
    row_map = lambda rows: {(r["video_id"], r["exp_id"]): r for r in rows}
    if dataset == "refyt":
        current = score_refyt(out)
        ref = row_map([{**r, "windows": r["scores"]["J"]} for r in current["rows"]])
        cand = row_map([{**r, "windows": r["scores"][LABEL]} for r in current["rows"]])
        assert current["means"]["Full"]["J"] == original_summary["means"]["Full"]["J"]
    elif dataset == "refsav":
        current = score_refsav(out)
        ref = row_map([{**r, "windows": {w: {"JF": x} for w, x in r["scores"]["J"].items()}} for r in current["rows"]])
        cand = row_map([{**r, "windows": {w: {"JF": x} for w, x in r["scores"][LABEL].items()}} for r in current["rows"]])
        assert current["results"]["overall"]["windows"]["Full"]["means_JF"]["J"] == original_summary["results"]["overall"]["windows"]["Full"]["means_JF"]["J"]
    elif dataset == "davis":
        _, adapter = runner(dataset, out)
        adapter["score"]()
        current = read(out / "summary.json")
        ref = row_map([{**r, "windows": {**{w: r[w] for w in WINDOWS if w != "Future"}, "Future": r["Future5"]}}
                       for r in original_summary["rows"]])
        cand = row_map([{**r, "windows": {**{w: r[w] for w in WINDOWS if w != "Future"}, "Future": r["Future5"]}}
                        for r in current["rows"]])
    else:
        history = out / "original_spa_windows"
        history.mkdir(exist_ok=True)
        archived = reference_windows(dataset)
        current = archived_score_with_windows(dataset, out)
        key = "means" if dataset == "mevis" else "official"
        archived_means = archived[key]["overall"] if dataset == "revos" else archived[key]
        reference_means = original_summary[key]["overall"] if dataset == "revos" else original_summary[key]
        assert archived_means.keys() == reference_means.keys()
        assert all(math.isclose(v, reference_means[m], rel_tol=0, abs_tol=1e-10)
                   for m, v in archived_means.items())
        ref, cand = row_map(archived["rows"]), row_map(current["rows"])
    comparisons = paired(ref, cand, by_type=dataset == "revos")
    if dataset == "revos":
        comparisons["Full"]["official"] = {
            "original_spa": original_summary["official"], LABEL: current["official"]}
    write(out / "results.json", {"dataset": dataset, "protocol_sha256": sha(out / "protocol.json"),
                                 "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
                                 "matched_expressions": len(ref), "comparison": comparisons,
                                 "revos_window_aggregation": "category-balanced without per-expression rounding" if dataset == "revos" else None})


def reference_windows(dataset):
    out = SUITE / dataset
    baseline = BASELINES[dataset]
    history = out / "original_spa_windows"
    history.mkdir(exist_ok=True)
    with (history / "scoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in ("protocol.json", "prediction_freeze.json", "predictions"):
            link = history / name
            if not link.exists():
                link.symlink_to(baseline / name)
            assert link.resolve() == (baseline / name).resolve()
        if (history / "summary.json").is_file():
            return read(history / "summary.json")
        return archived_score_with_windows(dataset, history)


def run():
    import evaluation_controller as controller

    # Existing UUID lock and ready/permit checks, restricted to the physical card requested.
    find_available = controller.available
    controller.available = lambda minimum: {k: v for k, v in find_available(minimum).items() if k == GPU}
    scoring = []
    failures = []
    with (SUITE / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dataset in ORDER:
            out = SUITE / dataset
            controller.HERE, controller.WORKER = out, out / WORKER.name
            controller.PROTOCOL = out / "protocol.json"
            controller.STATE = out / "evaluation_state.json"
            try:
                p = validate(out)
                if not (out / "prediction_freeze.json").exists():
                    controller.admit(p)
                verify_freeze(out)
                controller.state("scoring")
                with (out / "score.log").open("a") as log:
                    proc = subprocess.Popen([sys.executable, "-u", str(controller.WORKER), "score"],
                                            cwd=ROOT, env=controller.worker_env(""), stdout=log, stderr=subprocess.STDOUT)
                scoring.append((out, proc))
            except Exception as exc:
                controller.state("failed", reason=str(exc))
                failures.append(dataset)
                print(dataset, repr(exc), flush=True)
        for out, proc in scoring:
            if proc.wait():
                failures.append(out.name)
    write(SUITE / "suite_result.json", {"state": "failed" if failures else "complete", "failed": failures,
                                        "results": {d: sha(SUITE / d / "results.json") for d in ORDER
                                                    if (SUITE / d / "results.json").exists()}})


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "register":
        for dataset in ORDER:
            register(dataset)
    elif action == "run":
        run()
    elif action == "reference":
        reference_windows(sys.argv[2])
    elif action == "infer":
        infer(WORKER.parent)
    elif action == "score":
        out = WORKER.parent
        try:
            score(out)
        except Exception as exc:
            write(out / "evaluation_state.json", {"state": "failed", "reason": repr(exc)})
            raise
        write(out / "evaluation_state.json", {"state": "complete", "result_sha256": sha(out / "results.json")})
    else:
        raise ValueError(action)
