"""Build the feature dataset of the crossed design: runner, assembly and checks.

The dataset is one row per run and checkpoint with every column that
dataset_features.COLUMNS lists, and one row per run with the configuration,
the grokking times under several definitions and the values at the events.
It is built for exploratory analysis and model fitting on how the kernel
scale, rotation and alignment relate to each other and to the grokking time.

The design is the crossed grid of alpha, width and weight decay, with five
seeds per cell. Everything else is fixed as ntk_lib.load_cell fixes it for the
Tier 1 grid: NTK parameterisation, base learning rate 100, p = 23, training
fraction 0.9, data seed 42, relu, and the mixed probe of the report's Kernel
metrics section. The baseline cell, alpha 1, width 100 and no decay, is the
setup of Kumar et al. (2024), Appendix 8.3, arXiv v3, and the alpha slice at
width 100 without decay is the Tier 0 sweep. The width by decay part is the
Tier 2 grid of the report.

The decay levels differ from the report's Tier 2 line, which lists eta lambda
in {0, 1e-3, 1e-2, 1e-1}. The Tier 1 notebook found that 1e-3 and above
collapse the weights at width 100, so the levels here cover three decades
below that and two levels inside the collapse region. The manifest records
this.

The probe kernel is computed with ntk_lib.entk_closed_form, which matches the
autograd kernel to float32 rounding. Every run checks this at its first and
last checkpoint and saves the result.

Run it from repoduced-code with the environment at the repository root.

    uv run python dataset_sweep.py check        # closed form, Fourier test, logger leaves training unchanged
    uv run python dataset_sweep.py pilot        # timings under full load and the projected wall time
    uv run python dataset_sweep.py run          # the full sweep; it resumes where it stopped
    uv run python dataset_sweep.py status       # counts of finished runs
    uv run python dataset_sweep.py assemble     # the tables, the data dictionary and the manifest
    uv run python dataset_sweep.py validate     # the checks, also run by assemble

Output goes to model_fitting/data/ at the repository root, next to the
model fitting work that uses it. The per-run files in runs/ and the long
table checkpoints.parquet are ignored by git. runs.csv, runs.parquet, the
data dictionary and the manifest are committed. So is curves/, which holds
every checkpoint column of the seed 0 runs as one numpy file per run. The
animations environment has numpy but no pandas, and it reads those files
through animations/common/data.py.
"""

import argparse
import json
import logging
import logging.handlers
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "model_fitting" / "data"
RUNS = OUT / "runs"
LOGS = OUT / "logs"

ALPHAS = [0.5, 1.0, 2.0]
WIDTHS = [50, 100, 200, 400, 800, 1600]
DECAYS = [0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
SEEDS = [0, 1, 2, 3, 4]
STEPS = 200_000
LINEAR_INTERVAL = 1000
N_LOG = 120
# Weights are saved at step 0, at the last step, at this many log-spaced
# checkpoints, and every WEIGHT_EVERY steps.
N_WEIGHT_LOG, WEIGHT_EVERY = 40, 20_000
# Widths at or above this go to the pool of wide runs. Wide runs share the
# matrix unit of the processor, and throughput stops rising at about five of
# them at once.
WIDE = 800
WIDE_WORKERS, NARROW_WORKERS = 4, 8
# The seeds whose curves are written to curves/ and committed, so the
# animations can draw any cell without the ignored long table. Seed 0 of all
# 144 cells takes about 22 MB.
ANIMATION_SEEDS = (0,)

ACC_LEVELS = (0.95, 0.99, 1.0)
LOSS_TAUS = (3e-2, 2e-2, 1e-2, 3e-3, 1e-3)
A_LEVELS = (0.08, 0.10, 0.12, 0.14)
EVENT_FIELDS = ("A_t", "S_c", "R_c", "gamma", "D", "weight_norm", "A_full", "S_W1", "S_W2")
TRAINING_FIELDS = ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm", "param_dist")

DECAY_NOTE = ("The report's Tier 2 line lists eta lambda in {0, 1e-3, 1e-2, 1e-1}. The Tier 1 notebook found that "
              "eta lambda of 1e-3 and above collapses the weights at width 100 and alpha 1, so this dataset uses "
              "{0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}. The two largest levels sample the collapse region. "
              "The report was not edited.")


# =====================================================================
# Design
# =====================================================================
def checkpoint_grid(steps):
    """Step 0, every LINEAR_INTERVAL steps, and N_LOG log-spaced steps. It holds the last step."""
    import ntk_trace as T
    return T.checkpoint_grid(steps, interval=LINEAR_INTERVAL, n_log=N_LOG)


def log_steps(steps):
    """The log-spaced part of the grid, rounded to integers."""
    return sorted(set(np.rint(np.logspace(0, np.log10(steps), N_LOG)).astype(int).tolist()))


def weight_steps(steps, grid):
    """The checkpoints at which the weights are saved."""
    grid = np.asarray(grid)
    targets = np.logspace(0, np.log10(steps), N_WEIGHT_LOG)
    picked = {int(grid[np.argmin(np.abs(grid - t))]) for t in targets}
    return sorted(picked | {0, int(steps)} | set(range(0, steps + 1, WEIGHT_EVERY)))


def run_id(width, alpha, eta_kappa, seed):
    import ntk_lib as L
    return L.cell_name(width, alpha, eta_kappa, seed)


def run_config(width, alpha, eta_kappa, seed, steps):
    """The keyword arguments of ntk_lib.train_run for one run."""
    return dict(parameterisation="ntk", hidden_dim=width, alpha=alpha, eta_0=100.0, eta_kappa=eta_kappa,
                seed=seed, steps=steps, eval_interval=LINEAR_INTERVAL, kernel_save_interval=steps,
                checkpoint_steps=checkpoint_grid(steps), probe="mixed", kernel_method="closed_form")


def build_tasks(alphas, widths, decays, seeds, steps, out_dir, grid=None):
    grid_steps = grid if grid is not None else checkpoint_grid(steps)
    tasks = []
    for width in widths:
        for alpha in alphas:
            for ek in decays:
                for seed in seeds:
                    config = run_config(width, alpha, ek, seed, steps)
                    config["checkpoint_steps"] = list(grid_steps)
                    tasks.append(dict(run_id=run_id(width, alpha, ek, seed), width=width, config=config,
                                      save_steps=weight_steps(steps, grid_steps), log_steps=log_steps(steps),
                                      out_dir=str(out_dir)))
    return tasks


def interleave(tasks):
    """Order tasks round-robin over widths, widest first, so a pool always holds a mix of widths."""
    by_width = {}
    for t in tasks:
        by_width.setdefault(t["width"], []).append(t)
    queues = [by_width[w] for w in sorted(by_width, reverse=True)]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


# =====================================================================
# Small utilities
# =====================================================================
def clean(obj):
    """Return obj with numpy values made plain and NaN or infinite floats made None, for JSON."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", ".")))


def versions():
    import torch
    import pandas
    return dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                pandas=pandas.__version__, platform=platform.platform(), machine=platform.machine(),
                host=platform.node())


def event(kind, **fields):
    return {"event": dict(kind=kind, **fields)}


class JsonLines(logging.Formatter):
    def format(self, record):
        entry = dict(time=self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), level=record.levelname,
                     process=record.processName, message=record.getMessage())
        entry.update(clean(getattr(record, "event", {})))
        return json.dumps(entry)


def log_handlers(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    text = logging.Formatter("%(asctime)s %(levelname)-7s %(processName)-18s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_dir / "sweep.log")
    file_handler.setFormatter(text)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(text)
    jsonl = logging.FileHandler(log_dir / "events.jsonl")
    jsonl.setFormatter(JsonLines())
    return [file_handler, console, jsonl]


def init_worker(queue):
    """Set up a worker process: one thread, no display, and logging through the queue."""
    os.environ["MPLBACKEND"] = "Agg"
    import torch
    torch.set_num_threads(1)
    root = logging.getLogger()
    root.handlers = [logging.handlers.QueueHandler(queue)]
    root.setLevel(logging.INFO)


# =====================================================================
# One run
# =====================================================================
def closed_form_check(logger, weights, width):
    """Largest relative difference between the autograd and the closed-form probe kernels at the saved steps."""
    import torch
    import ntk_lib as L
    p = logger.p
    out = {}
    for step, (W1, W2) in weights.items():
        model = L.NTKMLP(2 * p, width, p)
        with torch.no_grad():
            model.W1.copy_(torch.from_numpy(W1))
            model.W2.copy_(torch.from_numpy(W2))
        Ka = L.entk(model, logger.x_probe)
        Kc = L.entk_closed_form(model, logger.x_probe)
        out[str(step)] = (torch.linalg.norm(Ka - Kc) / torch.linalg.norm(Ka)).item()
    return out


def run_events(run):
    """The accuracy events at test level 1 and the loss events at tau 1e-2, for the log."""
    import ntk_lib as L
    acc = L.accuracy_crossings(run, 1.0)
    loss = L.grokking_time(run, 1e-2, 1e-2)
    return dict(t_mem=acc["t_train"], t_gen=acc["t_test"], t_grok_acc=acc["t_grok"], censored_acc=acc["censored"],
                t_grok_loss=loss["t_grok"], censored_loss=loss["censored"])


def run_one(task):
    """Train one run with the feature logger and write its files. Never raises."""
    import pandas as pd
    import torch
    import ntk_lib as L
    import dataset_features as F

    log = logging.getLogger("sweep")
    rid, out_dir = task["run_id"], Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{rid}_meta.json"
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
            if (old.get("status") in ("ok", "diverged") and old.get("config") == clean(task["config"])
                    and old.get("feature_version") == F.FEATURE_VERSION):
                return dict(run_id=rid, status="skipped", width=task["width"], seconds=0.0)
        except (json.JSONDecodeError, OSError):
            pass

    cfg = L.normalise_checkpoints({**L.DEFAULTS, **task["config"]})
    c = task["config"]
    log.info(f"{rid}: start, width {c['hidden_dim']}, alpha {c['alpha']:g}, eta_kappa {c['eta_kappa']:g}, "
             f"seed {c['seed']}, {c['steps']} steps", extra=event("start", run_id=rid))
    t0 = time.time()
    error, run, logger = None, None, None
    try:
        logger = F.FeatureLogger(cfg, rid, save_steps=task["save_steps"], linear_interval=LINEAR_INTERVAL,
                                 log_steps=task["log_steps"])
        run = L.train_run(rid, verbose=False, on_checkpoint=logger, results_dir=out_dir, **task["config"])
        status = "diverged" if run["diverged"] else "ok"
    except Exception:
        error, status = traceback.format_exc(), "failed"
    seconds = time.time() - t0

    meta = dict(run_id=rid, status=status, error=error, config=task["config"], feature_version=F.FEATURE_VERSION,
                seconds=seconds, git=git_state(), versions=versions(),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        if logger is not None and logger.rows:
            frame = pd.DataFrame(logger.rows, columns=F.COLUMN_NAMES)
            tmp = out_dir / f"{rid}.parquet.tmp"
            frame.to_parquet(tmp, index=False)
            os.replace(tmp, out_dir / f"{rid}.parquet")
            weights = dict(logger.weights)
            weights[logger.last_step] = logger.last_weights
            steps = sorted(weights)
            np.savez_compressed(out_dir / f"{rid}_weights.npz", steps=np.asarray(steps),
                                W1=np.stack([weights[s][0] for s in steps]),
                                W2=np.stack([weights[s][1] for s in steps]))
            ends = {s: weights[s] for s in (steps[0], steps[-1])}
            meta.update(n_checkpoints=len(logger.rows), last_step=logger.last_step,
                        callback_seconds=logger.callback_seconds,
                        closed_form_check=closed_form_check(logger, ends, c["hidden_dim"]),
                        identity_checks=F.identity_checks(logger.check_arrays()))
        if run is not None:
            meta.update(diverged_step=run["diverged_step"], events=run_events(run))
    except Exception:
        meta["status"] = "failed"
        meta["error"] = (meta["error"] or "") + traceback.format_exc()
    write_atomic(meta_path, json.dumps(clean(meta), indent=1))

    if meta["status"] == "failed":
        log.error(f"{rid}: failed after {seconds:.0f} s\n{meta['error']}", extra=event("failed", run_id=rid))
    else:
        e = meta.get("events", {})
        worst = max(meta.get("closed_form_check", {}).values(), default=float("nan"))
        grok = ("censored" if e.get("censored_acc") else f"t_grok {e.get('t_grok_acc')}") \
            if e.get("t_mem") is not None else "never memorised"
        log.info(f"{rid}: {meta['status']} in {seconds:.0f} s, {meta.get('n_checkpoints')} checkpoints, "
                 f"memorised at {e.get('t_mem')}, generalised at {e.get('t_gen')}, {grok}, "
                 f"closed-form check {worst:.1e}", extra=event("finish", run_id=rid, status=meta["status"],
                                                               seconds=seconds, **e))
        if worst > 1e-4:
            log.warning(f"{rid}: closed-form kernel differs from autograd by {worst:.1e}",
                        extra=event("closed_form_mismatch", run_id=rid, value=worst))
        if meta["status"] == "diverged":
            log.warning(f"{rid}: diverged at step {meta.get('diverged_step')}",
                        extra=event("diverged", run_id=rid, step=meta.get("diverged_step")))
    return dict(run_id=rid, status=meta["status"], width=task["width"], seconds=seconds)


def run_pools(tasks, wide_workers, narrow_workers, log):
    """Run the tasks on two spawn pools, wide and narrow, and log progress until all finish."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    listener = logging.handlers.QueueListener(queue, *logging.getLogger().handlers, respect_handler_level=True)
    listener.start()
    wide = interleave([t for t in tasks if t["width"] >= WIDE])
    narrow = interleave([t for t in tasks if t["width"] < WIDE])
    results, t_start = [], time.time()

    def done(result):
        results.append(result)

    def failed(exc):
        log.error(f"a worker raised: {exc!r}")
        results.append(dict(run_id="?", status="failed", width=0, seconds=0.0))

    pools = []
    try:
        for group, n in ((wide, wide_workers), (narrow, narrow_workers)):
            if not group:
                continue
            pool = ctx.Pool(min(n, len(group)), initializer=init_worker, initargs=(queue,), maxtasksperchild=20)
            pools.append(pool)
            for t in group:
                pool.apply_async(run_one, (t,), callback=done, error_callback=failed)
        total, last = len(tasks), 0.0
        while len(results) < total:
            time.sleep(5)
            if time.time() - last >= 60 or len(results) == total:
                last = time.time()
                counts = {s: sum(r["status"] == s for r in results) for s in ("ok", "diverged", "failed", "skipped")}
                trained = counts["ok"] + counts["diverged"] + counts["failed"]
                elapsed = time.time() - t_start
                rate = trained / elapsed * 3600 if trained else 0.0
                remaining = total - len(results)
                eta = remaining / (trained / elapsed) if trained else float("nan")
                log.info(f"progress: {len(results)}/{total} done ({counts['ok']} ok, {counts['diverged']} diverged, "
                         f"{counts['failed']} failed, {counts['skipped']} skipped), {rate:.0f} runs per hour, "
                         f"elapsed {elapsed / 3600:.2f} h, ETA {eta / 3600:.2f} h",
                         extra=event("progress", done=len(results), total=total, **counts))
        for pool in pools:
            pool.close()
        for pool in pools:
            pool.join()
    except KeyboardInterrupt:
        log.warning("interrupted; terminating the workers. Run again to resume.")
        for pool in pools:
            pool.terminate()
        raise
    finally:
        listener.stop()
    return results


# =====================================================================
# Subcommands: check, pilot, run, status
# =====================================================================
def gromov_weights(p, n_units, rng):
    """Weights of the exact solution of Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12.

    Unit k has W1 = (cos(2 pi f n1 / p + phi1), cos(2 pi f n2 / p + phi2)) and
    W2[q] = cos(-2 pi f q / p - phi3) with phi3 = phi1 + phi2, where f is the
    unit's frequency. The frequencies cycle through 1 to (p - 1) / 2.
    """
    n = np.arange(p)
    freqs = 1 + np.arange(n_units) % ((p - 1) // 2)
    phi1, phi2 = rng.uniform(-np.pi, np.pi, n_units), rng.uniform(-np.pi, np.pi, n_units)
    phi3 = phi1 + phi2
    W1 = np.concatenate([np.cos(2 * np.pi * freqs[:, None] * n / p + phi1[:, None]),
                         np.cos(2 * np.pi * freqs[:, None] * n / p + phi2[:, None])], axis=1)
    W2 = np.cos(-2 * np.pi * freqs[None, :] * n[:, None] / p - phi3[None, :])
    return W1, W2


def cmd_check(args, log):
    """The closed-form kernel against autograd, the Fourier measures on an exact solution, and the logger's effect."""
    import torch
    import ntk_lib as L
    import dataset_features as F
    from RepoducedCode import compute_entk

    torch.manual_seed(0)
    p = 23
    (Xtr, ytr), (Xte, yte) = L.make_modular_addition_dataset(p=p)
    x, _, _ = L.select_probe(Xtr, ytr, Xte, yte, "mixed", 256)
    worst, ok = 0.0, True
    for par in ("ntk", "mean_field"):
        for width in WIDTHS:
            for moved in (False, True):
                model = L.build_model(par, 2 * p, width, p)
                if moved:
                    with torch.no_grad():
                        for q in model.parameters():
                            q.add_(0.3 * torch.randn_like(q))
                Ka, Kc = L.entk(model, x), L.entk_closed_form(model, x)
                d = (torch.linalg.norm(Ka - Kc) / torch.linalg.norm(Ka)).item()
                params = dict(model.named_parameters())
                Fa = compute_entk(model, params, x, mode="first_logit").detach() if width <= 400 else None
                F1, F2 = L.closed_form_terms(model.W1, model.W2, x, *L.closed_form_scales(model), output=0)
                dfl = (torch.linalg.norm(Fa.double() - (F1 + F2)) / torch.linalg.norm(Fa.double())).item() \
                    if Fa is not None else float("nan")
                worst = max(worst, d, 0.0 if math.isnan(dfl) else dfl)
                log.info(f"check: {par:>10} width {width:5d} {'moved' if moved else 'init '}: trace kernel "
                         f"{d:.1e}, first-logit kernel {dfl:.1e}")
    ok &= worst < 1e-5
    log.info(f"check: largest relative difference {worst:.1e} ({'pass' if worst < 1e-5 else 'FAIL'}, limit 1e-5)")

    W1, W2 = gromov_weights(p, 44, np.random.default_rng(0))
    four = F.fourier_features(W1, W2, p)
    gromov_ok = all(abs(four[k] - 1) < 1e-9 for k in ("a_topshare", "b_topshare", "c_topshare", "match", "phase"))
    ok &= gromov_ok
    log.info(f"check: Fourier measures on the exact solution of Gromov (2023): "
             f"{ {k: round(v, 12) for k, v in four.items()} } ({'pass' if gromov_ok else 'FAIL'})")

    # The logger must leave training unchanged.
    scratch = OUT / "check"
    steps = 3000
    cfg = run_config(100, 1.0, 1e-4, 1, steps)
    cfg["checkpoint_steps"] = checkpoint_grid(steps)
    plain = L.train_run("plain", verbose=False, results_dir=scratch, **cfg)
    full = L.normalise_checkpoints({**L.DEFAULTS, **cfg})
    logger = F.FeatureLogger(full, "logged", save_steps=[0], log_steps=log_steps(steps))
    logged = L.train_run("logged", verbose=False, on_checkpoint=logger, results_dir=scratch, **cfg)
    diff = max(float(np.max(np.abs(np.asarray(plain["history"][k]) - np.asarray(logged["history"][k]))))
               for k in L.HISTORY_KEYS)
    ok &= diff == 0.0
    log.info(f"check: history with and without the logger differs by {diff} over {len(plain['history']['step'])} "
             f"checkpoints ({'pass' if diff == 0 else 'FAIL'})")
    log.info("check: all passed" if ok else "check: SOME CHECKS FAILED")
    return 0 if ok else 1


def cmd_pilot(args, log):
    """Short runs at every width under the real concurrency, and the projected wall time of the sweep."""
    out_dir = OUT / "pilot"
    steps = 3000
    grid = list(range(0, 1001, 50)) + [steps]
    tasks = []
    for width in WIDTHS:
        for alpha, seed in ((1.0, 0), (2.0, 1)):
            config = run_config(width, alpha, 1e-4, seed, steps)
            config["checkpoint_steps"] = grid
            tasks.append(dict(run_id=f"pilot_{run_id(width, alpha, 1e-4, seed)}", width=width, config=config,
                              save_steps=[0, steps], log_steps=[], out_dir=str(out_dir)))
    for t in tasks:
        meta = out_dir / f"{t['run_id']}_meta.json"
        if meta.exists():
            meta.unlink()
    run_pools(tasks, args.wide_workers, args.narrow_workers, log)

    import pandas as pd
    per_width = {}
    for t in tasks:
        frame = pd.read_parquet(out_dir / f"{t['run_id']}.parquet")
        wall = dict(zip(frame["step"], frame["wall_seconds"]))
        # From step 0 to 1000 there are 1000 steps and 20 checkpoints; from 1000 to 3000, 2000 steps and 1.
        A = np.array([[1000.0, 20.0], [2000.0, 1.0]])
        b = np.array([wall[1000] - wall[0], wall[steps] - wall[1000]])
        per_width.setdefault(t["width"], []).append(np.linalg.solve(A, b))
    n_ckpt = len(checkpoint_grid(args.steps))
    runs_per_width = len(args.alphas) * len(args.decays) * len(args.seeds)
    wide_total = narrow_total = 0.0
    log.info(f"pilot: {n_ckpt} checkpoints per run of {args.steps} steps, {runs_per_width} runs per width")
    for width in sorted(per_width):
        step_s, ckpt_s = np.mean(per_width[width], axis=0)
        per_run = args.steps * step_s + n_ckpt * ckpt_s
        if width >= WIDE:
            wide_total += per_run * runs_per_width
        else:
            narrow_total += per_run * runs_per_width
        log.info(f"pilot: width {width:5d}: {step_s * 1000:.3f} ms per step, {ckpt_s * 1000:.0f} ms per checkpoint, "
                 f"{per_run / 60:.1f} min per run")
    wide_wall, narrow_wall = wide_total / args.wide_workers, narrow_total / args.narrow_workers
    log.info(f"pilot: projected wall time {max(wide_wall, narrow_wall) / 3600:.2f} h "
             f"(wide pool {wide_wall / 3600:.2f} h on {args.wide_workers} workers, narrow pool "
             f"{narrow_wall / 3600:.2f} h on {args.narrow_workers} workers). The rates were measured with both pools "
             f"full, so the narrow pool will run faster once the wide pool finishes, and the other way round.")
    return 0


def cmd_run(args, log):
    tasks = build_tasks(args.alphas, args.widths, args.decays, args.seeds, args.steps, RUNS)
    with open(LOGS / "commands.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, git=git_state())) + "\n")
    log.info(f"run: {len(tasks)} runs, {args.wide_workers} wide and {args.narrow_workers} narrow workers, "
             f"{len(checkpoint_grid(args.steps))} checkpoints per run, output in {RUNS}",
             extra=event("sweep_start", n_runs=len(tasks)))
    results = run_pools(tasks, args.wide_workers, args.narrow_workers, log)
    counts = {s: sum(r["status"] == s for r in results) for s in ("ok", "diverged", "failed", "skipped")}
    log.info(f"run: finished, {counts}", extra=event("sweep_finish", **counts))
    return 0 if counts["failed"] == 0 else 1


def cmd_status(args, log):
    metas = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*_meta.json"))]
    total = len(build_tasks(args.alphas, args.widths, args.decays, args.seeds, args.steps, RUNS))
    counts = {}
    for m in metas:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    print(f"{len(metas)} of {total} runs have finished: {counts}")
    by_width = {}
    for m in metas:
        by_width.setdefault(m["config"]["hidden_dim"], []).append(m["seconds"])
    for width in sorted(by_width):
        s = by_width[width]
        print(f"  width {width:5d}: {len(s):4d} runs, mean {np.mean(s):.0f} s per run")
    return 0


# =====================================================================
# Assembly
# =====================================================================
RUN_COLUMNS = []


def rcol(name, meaning, source="-"):
    RUN_COLUMNS.append(dict(name=name, group="run", meaning=meaning, formula="-", source=source, units="-"))


def acc_tag(level):
    return f"{round(level * 100):d}"


def loss_tag(tau):
    return f"{tau:.0e}".replace("-0", "-")


def a_tag(level):
    return f"{level:.2f}".replace("0.", "")


rcol("run_id", "The name of the run, the grid cell name of ntk_lib.cell_name.", "ntk_lib.cell_name")
rcol("alpha", "The laziness knob alpha.", "Kumar et al. (2024), Appendix 8.1, Equation 7, arXiv v3")
rcol("width", "The hidden width N.")
rcol("eta_kappa", "The weight decay eta lambda.", "docs/report.tex, The laziness and weight-decay knobs")
rcol("seed", "The model seed.")
rcol("lr", "The learning rate eta_0 / alpha^2.", "ntk_lib.train_run")
rcol("weight_decay", "The decay coefficient eta_kappa / lr passed to torch.optim.SGD.", "ntk_lib.train_run")
rcol("status", "ok, diverged (the training loss stopped being finite) or failed (an exception).")
rcol("error", "The traceback of a failed run, empty otherwise.")
rcol("diverged_step", "The checkpoint at which the training loss was first not finite.")
rcol("steps", "The step budget.")
rcol("last_step", "The last recorded checkpoint.")
rcol("n_checkpoints", "The number of recorded checkpoints.")
rcol("seconds", "Wall-clock seconds for the run, logger included.")
rcol("callback_seconds", "Wall-clock seconds spent in the feature logger.")
rcol("feature_version", "dataset_features.FEATURE_VERSION when the run was made.")
rcol("git_commit", "The commit of the repository when the run was made.")
rcol("git_dirty", "True if repoduced-code had uncommitted changes when the run was made.")
rcol("closed_form_check", "Largest relative difference between the autograd and closed-form probe kernels at the "
     "first and last checkpoints.", "ntk_lib.entk_closed_form")
rcol("init_weight_norm", "Weight norm at step 0.", "ntk_lib.train_run")
rcol("final_weight_norm", "Weight norm at the last checkpoint.")
rcol("weight_norm_growth", "final_weight_norm / init_weight_norm.")
for name in ("A_t", "A_full", "S_c", "R_c", "train_acc", "test_acc", "train_loss", "test_loss"):
    rcol(f"{name}_final", f"{name} at the last checkpoint.")
rcol("A_0", "A_t at step 0.")
rcol("A_full_0", "A_full at step 0.")
rcol("A_peak", "Largest A_t over the run.")
rcol("A_peak_step", "The step of A_peak.")
rcol("memorised", "True if the training accuracy reached 1 at some checkpoint. A run that never memorises has no "
     "grokking time, which is different from a censored one.")
rcol("t_mem", "First checkpoint at which the training accuracy is 1.", "ntk_lib.accuracy_crossings")
rcol("t_mem_prev", "The checkpoint before t_mem, so the event lies in (t_mem_prev, t_mem].")
for level in ACC_LEVELS:
    tag = acc_tag(level)
    rcol(f"t_gen_acc{tag}", f"First checkpoint at which the test accuracy reaches {level:g}.",
         "ntk_lib.accuracy_crossings")
    rcol(f"t_gen_acc{tag}_prev", f"The checkpoint before t_gen_acc{tag}.")
    rcol(f"t_grok_acc{tag}", f"Grokking time under the accuracy definition at test level {level:g}: t_gen_acc{tag} - "
         "t_mem, or last_step - t_mem if censored. Empty if the run never memorised.", "ntk_lib.accuracy_crossings")
    rcol(f"censored_acc{tag}", f"True if the test accuracy never reached {level:g}, or the run never memorised.",
         "ntk_lib.accuracy_crossings")
for tau in LOSS_TAUS:
    tag = loss_tag(tau)
    rcol(f"t_train_loss{tag}", f"First checkpoint at which the training loss is below {tau:g}.",
         "ntk_lib.grokking_time")
    rcol(f"t_test_loss{tag}", f"First checkpoint at which the test loss is below {tau:g}.", "ntk_lib.grokking_time")
    rcol(f"t_grok_loss{tag}", f"Grokking time under the pre-registered loss definition with both thresholds {tau:g}.",
         "ntk_lib.grokking_time; docs/report.tex, Defining t_grok")
    rcol(f"censored_loss{tag}", f"True if the test loss never fell below {tau:g}, or the training loss never did.",
         "ntk_lib.grokking_time")
for when in ("mem", "gen"):
    word = "t_mem" if when == "mem" else "t_gen_acc100"
    for name in EVENT_FIELDS:
        rcol(f"{name}_at_{when}", f"{name} at {word}. Empty if the event did not happen.")
for level in A_LEVELS:
    rcol(f"t_A{a_tag(level)}", f"First checkpoint at which A_t reaches {level:g}. Empty if it never does.")


def first_at_least(steps, values, level):
    idx = np.nonzero(np.asarray(values) >= level)[0]
    return int(steps[idx[0]]) if len(idx) else None


def previous_step(steps, t):
    if t is None:
        return None
    i = int(np.searchsorted(steps, t))
    return int(steps[i - 1]) if i > 0 else None


def run_row(meta, frame, out_dir):
    """One row of the runs table, from a run's meta file, its feature rows and its train_run JSON."""
    import ntk_lib as L
    c = meta["config"]
    lr = c["eta_0"] / c["alpha"] ** 2
    row = dict(run_id=meta["run_id"], alpha=c["alpha"], width=c["hidden_dim"], eta_kappa=c["eta_kappa"],
               seed=c["seed"], lr=lr, weight_decay=c["eta_kappa"] / lr, status=meta["status"],
               error=meta.get("error") or "", diverged_step=meta.get("diverged_step"), steps=c["steps"],
               last_step=meta.get("last_step"), n_checkpoints=meta.get("n_checkpoints"), seconds=meta["seconds"],
               callback_seconds=meta.get("callback_seconds"), feature_version=meta["feature_version"],
               git_commit=meta["git"]["commit"], git_dirty=meta["git"]["dirty"],
               closed_form_check=max((v for v in meta.get("closed_form_check", {}).values() if v is not None),
                                     default=None))
    path = out_dir / f"{meta['run_id']}.json"
    if meta["status"] == "failed" or frame is None or not path.exists():
        return row
    run = json.loads(path.read_text())
    h = run["history"]
    steps = np.asarray(h["step"])
    f = frame.set_index("step")
    row.update(init_weight_norm=run["init_weight_norm"], final_weight_norm=h["weight_norm"][-1],
               weight_norm_growth=h["weight_norm"][-1] / run["init_weight_norm"])
    for name in ("A_t", "A_full", "S_c", "R_c", "train_acc", "test_acc", "train_loss", "test_loss"):
        row[f"{name}_final"] = float(f[name].iloc[-1])
    row.update(A_0=float(f["A_t"].iloc[0]), A_full_0=float(f["A_full"].iloc[0]),
               A_peak=float(f["A_t"].max()), A_peak_step=int(f["A_t"].idxmax()))
    acc = {level: L.accuracy_crossings(run, level) for level in ACC_LEVELS}
    t_mem = acc[1.0]["t_train"]
    row.update(memorised=t_mem is not None, t_mem=t_mem, t_mem_prev=previous_step(steps, t_mem))
    for level, g in acc.items():
        tag = acc_tag(level)
        row[f"t_gen_acc{tag}"] = g["t_test"]
        row[f"t_gen_acc{tag}_prev"] = previous_step(steps, g["t_test"])
        row[f"t_grok_acc{tag}"] = g["t_grok"]
        row[f"censored_acc{tag}"] = g["censored"]
    for tau in LOSS_TAUS:
        g = L.grokking_time(run, tau, tau)
        tag = loss_tag(tau)
        row.update({f"t_train_loss{tag}": g["t_train"], f"t_test_loss{tag}": g["t_test"],
                    f"t_grok_loss{tag}": g["t_grok"], f"censored_loss{tag}": g["censored"]})
    for when, t in (("mem", t_mem), ("gen", acc[1.0]["t_test"])):
        for name in EVENT_FIELDS:
            row[f"{name}_at_{when}"] = float(f.loc[t, name]) if t is not None else None
    for level in A_LEVELS:
        row[f"t_A{a_tag(level)}"] = first_at_least(f.index.values, f["A_t"].values, level)
    return row


def write_curves(ckpt, out):
    """Write the checkpoint columns of the runs with a seed in ANIMATION_SEEDS, one numpy file per run.

    Each file curves/<run_id>.npz holds one array per column of
    checkpoints.parquet, in checkpoint order. step is int64, the flags are
    bool, and every other series is float32, which is enough for a picture.
    alpha, width, eta_kappa and seed are stored once as 0-d arrays. A file
    left over from an earlier assembly whose run is no longer selected is
    removed, so the folder always matches the table.
    """
    import dataset_features as F
    curves = out / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    ids = ("alpha", "width", "eta_kappa", "seed")
    written = set()
    for rid, frame in ckpt[ckpt["seed"].isin(ANIMATION_SEEDS)].groupby("run_id", sort=False):
        frame = frame.sort_values("step")
        arrays = {}
        for name in F.COLUMN_NAMES:
            if name == "run_id":
                continue
            values = frame[name].to_numpy()
            if name in ids:
                arrays[name] = np.asarray(values[0])
            elif name == "step":
                arrays[name] = values.astype(np.int64)
            elif values.dtype == bool:
                arrays[name] = values
            else:
                arrays[name] = values.astype(np.float32)
        np.savez_compressed(curves / f"{rid}.npz", **arrays)
        written.add(f"{rid}.npz")
    for stale in curves.glob("*.npz"):
        if stale.name not in written:
            stale.unlink()
    return len(written)


def write_dictionary(out):
    import pandas as pd
    import dataset_features as F
    entries = [dict(table="checkpoints", **c) for c in F.COLUMNS] + [dict(table="runs", **c) for c in RUN_COLUMNS]
    frame = pd.DataFrame(entries, columns=["table", "name", "group", "meaning", "formula", "source", "units"])
    frame.to_csv(out / "data_dictionary.csv", index=False)
    lines = ["# Data dictionary", "",
             "This file is written by `dataset_sweep.py assemble` from `dataset_features.COLUMNS` and "
             "`dataset_sweep.RUN_COLUMNS`. Edit those, not this file.", "",
             "`checkpoints.parquet` has one row per run and checkpoint. `runs.parquet` and `runs.csv` have one row "
             "per run. The two tables join on `run_id`. The weights at about 50 checkpoints per run are in "
             "`runs/<run_id>_weights.npz`, with arrays `steps`, `W1` of shape (T, N, 2p) and `W2` of shape "
             "(T, p, N). `ntk_lib.entk_closed_form` rebuilds any kernel from them.", "",
             f"`curves/<run_id>.npz` holds every checkpoint column of the runs with seed in {list(ANIMATION_SEEDS)}, "
             "one array per column, for the animations environment, which has numpy but no pandas. "
             "`animations/common/data.py` loads them. Unlike `checkpoints.parquet` and `runs/`, these files are "
             "committed.", "",
             "Notation. H = I - (1/n) 1 1^T. Kc = H K H is the centred kernel and k = Kc / ||Kc||_F the unit "
             "centred kernel. Y is the one-hot label matrix and G = H Y Y^T H. <A, B> is the Frobenius inner "
             "product. The probe is the mixed probe of 203 training and 53 test pairs.", ""]
    for table in ("checkpoints", "runs"):
        lines += [f"## {table}", ""]
        sub = frame[frame["table"] == table]
        for group, rows in sub.groupby("group", sort=False):
            lines += [f"### {group}", "", "| name | meaning | formula | source |", "| --- | --- | --- | --- |"]
            for _, r in rows.iterrows():
                cells = [str(r[k]).replace("|", "\\|") for k in ("name", "meaning", "formula", "source")]
                lines.append("| `" + cells[0] + "` | " + " | ".join(cells[1:]) + " |")
            lines.append("")
    (out / "data_dictionary.md").write_text("\n".join(lines))
    return frame


def cmd_assemble(args, log):
    import pandas as pd
    import dataset_features as F

    metas = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*_meta.json"))]
    if not metas:
        log.error(f"assemble: no finished runs in {RUNS}")
        return 1
    frames, rows = [], []
    for m in metas:
        path = RUNS / f"{m['run_id']}.parquet"
        frame = pd.read_parquet(path) if m["status"] != "failed" and path.exists() else None
        if frame is not None:
            frames.append(frame)
        rows.append(run_row(m, frame, RUNS))
    ckpt = pd.concat(frames, ignore_index=True).sort_values(["width", "alpha", "eta_kappa", "seed", "step"])
    undocumented = set(ckpt.columns) - set(F.COLUMN_NAMES)
    if undocumented or list(ckpt.columns) != F.COLUMN_NAMES:
        raise AssertionError(f"checkpoint columns do not match dataset_features.COLUMNS: {sorted(undocumented)}")
    ckpt.to_parquet(OUT / "checkpoints.parquet", index=False)
    runs = pd.DataFrame(rows, columns=[c["name"] for c in RUN_COLUMNS])
    extra = set(pd.DataFrame(rows).columns) - set(runs.columns)
    if extra:
        raise AssertionError(f"run columns missing from RUN_COLUMNS: {sorted(extra)}")
    runs = runs.sort_values(["width", "alpha", "eta_kappa", "seed"])
    runs.to_parquet(OUT / "runs.parquet", index=False)
    runs.to_csv(OUT / "runs.csv", index=False)
    write_dictionary(OUT)
    n_curves = write_curves(ckpt, OUT)

    ok = runs[runs["status"] != "failed"]
    per_cell = []
    for (w, a, ek), g in ok.groupby(["width", "alpha", "eta_kappa"]):
        per_cell.append(dict(width=int(w), alpha=float(a), eta_kappa=float(ek), runs=len(g),
                             never_memorised=int((~g["memorised"].astype(bool)).sum()),
                             grokked_acc100=int((g["memorised"].astype(bool) & ~g["censored_acc100"].astype(bool)).sum()),
                             censored_acc100=int((g["memorised"].astype(bool) & g["censored_acc100"].astype(bool)).sum())))
    design = dict(alphas=sorted(runs["alpha"].unique().tolist()), widths=sorted(runs["width"].unique().tolist()),
                  eta_kappas=sorted(runs["eta_kappa"].unique().tolist()), seeds=sorted(runs["seed"].unique().tolist()),
                  steps=sorted(runs["steps"].unique().tolist()))
    manifest = dict(
        written=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, git=git_state(), versions=versions(),
        design=design, fixed=dict(parameterisation="ntk", eta_0=100.0, p=23, train_fraction=0.9, data_seed=42,
                                  activation="relu", init_scale=1.0, probe="mixed", probe_size=256,
                                  kernel_method="closed_form", linear_interval=LINEAR_INTERVAL, n_log=N_LOG),
        feature_version=F.FEATURE_VERSION, krr_ridge=F.KRR_RIDGE, sharpness_iters=F.SHARPNESS_ITERS,
        acc_levels=ACC_LEVELS, loss_taus=LOSS_TAUS, a_levels=A_LEVELS,
        counts=dict(runs=len(runs), rows=len(ckpt), columns=len(ckpt.columns),
                    status=runs["status"].value_counts().to_dict()),
        total_seconds=float(runs["seconds"].sum()), per_cell=per_cell, decay_levels_note=DECAY_NOTE,
        files=dict(checkpoints="checkpoints.parquet", runs=["runs.parquet", "runs.csv"],
                   dictionary=["data_dictionary.csv", "data_dictionary.md"], weights="runs/<run_id>_weights.npz",
                   curves=f"curves/<run_id>.npz, {n_curves} runs with seed in {list(ANIMATION_SEEDS)}, committed",
                   run_json="runs/<run_id>.json", logs="logs/sweep.log, logs/events.jsonl"))
    write_atomic(OUT / "manifest.json", json.dumps(clean(manifest), indent=1))
    log.info(f"assemble: {len(runs)} runs, {len(ckpt)} checkpoint rows x {len(ckpt.columns)} columns, "
             f"{(OUT / 'checkpoints.parquet').stat().st_size / 1e6:.0f} MB; status {manifest['counts']['status']}; "
             f"{n_curves} curve files for the animations")
    return cmd_validate(args, log)


# =====================================================================
# Validation
# =====================================================================
def cmd_validate(args, log):
    import ntk_lib as L

    metas = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*_meta.json"))]
    good = [m for m in metas if m["status"] != "failed"]
    out = {}

    cf = [max(v for v in m["closed_form_check"].values() if v is not None) for m in good
          if m.get("closed_form_check") and any(v is not None for v in m["closed_form_check"].values())]
    out["closed_form"] = dict(runs=len(cf), largest=max(cf, default=None), limit=1e-4,
                              passed=bool(cf) and max(cf) < 1e-4)

    names = sorted({k for m in good for k in m.get("identity_checks", {})})
    worst = {k: max((m["identity_checks"][k] for m in good if m.get("identity_checks", {}).get(k) is not None),
                    default=None) for k in names}
    out["identities"] = worst

    # Reproduction against the saved grid runs, which were trained with the autograd kernel.
    compared, exact, kernel_worst, mismatches = 0, 0, 0.0, []
    for m in good:
        saved_path = L.RESULTS / f"{m['run_id']}.json"
        mine_path = RUNS / f"{m['run_id']}.json"
        if not saved_path.exists() or not mine_path.exists():
            continue
        saved, mine = json.loads(saved_path.read_text())["history"], json.loads(mine_path.read_text())["history"]
        index = {s: i for i, s in enumerate(mine["step"])}
        shared = [(index[s], j) for j, s in enumerate(saved["step"]) if s in index]
        if not shared:
            continue
        compared += 1
        first_bad, largest = None, 0.0
        for i, j in shared:
            d = max(abs(mine[k][i] - saved[k][j]) for k in TRAINING_FIELDS)
            largest = max(largest, d)
            if d != 0 and first_bad is None:
                first_bad = saved["step"][j]
        if largest == 0:
            exact += 1
        else:
            mismatches.append(dict(run_id=m["run_id"], largest=largest, first_step=first_bad))
        for k in ("S_c", "R_c", "A_t", "A_u", "S_t", "R_t", "K_norm"):
            pairs = np.array([(mine[k][i], saved[k][j]) for i, j in shared], dtype=float)
            scale = max(float(np.nanmax(np.abs(saved[k]))), 1e-30)
            kernel_worst = max(kernel_worst, float(np.nanmax(np.abs(pairs[:, 0] - pairs[:, 1]))) / scale)
    out["reproduction"] = dict(runs_compared=compared, exact_in_training_fields=exact, mismatches=mismatches,
                               kernel_fields_largest_relative=kernel_worst,
                               note="Training fields must agree exactly. Kernel fields differ at float32 rounding "
                                    "because the saved runs used the autograd kernel.")

    # The Tier 0 slice: width 100, no decay. The new grid has log-spaced checkpoints the saved grid lacks, so
    # both histories are cut to the checkpoints they share before the crossings are found.
    def restrict(run, keep):
        h = run["history"]
        idx = [i for i, s in enumerate(h["step"]) if s in keep]
        return dict(history={k: [h[k][i] for i in idx] for k in ("step", "train_acc", "test_acc")})

    tier0 = []
    for m in good:
        c = m["config"]
        if c["hidden_dim"] != 100 or c["eta_kappa"] != 0.0:
            continue
        saved_path = L.RESULTS / f"{m['run_id']}.json"
        if not saved_path.exists():
            continue
        saved_run = json.loads(saved_path.read_text())
        mine_run = json.loads((RUNS / f"{m['run_id']}.json").read_text())
        keep = set(saved_run["history"]["step"]) & set(mine_run["history"]["step"])
        saved = L.accuracy_crossings(restrict(saved_run, keep), 1.0)
        mine = L.accuracy_crossings(restrict(mine_run, keep), 1.0)
        agree = saved["t_train"] == mine["t_train"] and saved["t_test"] == mine["t_test"]
        tier0.append(dict(run_id=m["run_id"], shared_checkpoints=len(keep), saved=saved, mine=mine, agree=agree))
    out["tier0_slice"] = dict(runs=len(tier0), agree=sum(r["agree"] for r in tier0),
                              disagreements=[r for r in tier0 if not r["agree"]])

    write_atomic(OUT / "validation.json", json.dumps(clean(out), indent=1))
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["validation"] = clean(out)
        write_atomic(manifest_path, json.dumps(manifest, indent=1))
    log.info(f"validate: closed form largest {out['closed_form']['largest']} over {len(cf)} runs; "
             f"reproduction {exact}/{compared} runs exact in training fields, kernel fields within "
             f"{kernel_worst:.1e}; Tier 0 slice {out['tier0_slice']['agree']}/{len(tier0)} agree")
    for k, v in worst.items():
        log.info(f"validate: {k:>45}: largest {v}")
    return 0


# =====================================================================
# Entry point
# =====================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["check", "pilot", "run", "status", "assemble", "validate"])
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument("--decays", type=float, nargs="+", default=DECAYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--wide-workers", type=int, default=WIDE_WORKERS)
    parser.add_argument("--narrow-workers", type=int, default=NARROW_WORKERS)
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory instead of model_fitting/data, for smoke tests")
    args = parser.parse_args(argv)
    if args.out is not None:
        global OUT, RUNS, LOGS
        OUT = args.out.resolve()
        RUNS, LOGS = OUT / "runs", OUT / "logs"

    root = logging.getLogger()
    root.handlers = log_handlers(LOGS)
    root.setLevel(logging.INFO)
    log = logging.getLogger("sweep")
    commands = dict(check=cmd_check, pilot=cmd_pilot, run=cmd_run, status=cmd_status, assemble=cmd_assemble,
                    validate=cmd_validate)
    return commands[args.command](args, log)


if __name__ == "__main__":
    sys.exit(main())
