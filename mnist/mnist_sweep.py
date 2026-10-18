"""The MNIST grid sweep: runner, assembly and checks.

This is the MNIST counterpart of repoduced-code/dataset_sweep.py, Zain's sweep
on modular addition, and it writes the same kinds of files. The dataset is
one row per run and checkpoint with every column that mnist_features.COLUMNS
lists, and one row per run with the configuration, the grokking times under
several definitions and the values at the events.

The design is the crossed grid of initial scale, width and weight decay, with
five seeds per cell. Everything else is fixed at the setup the exploration
found to grok (exploration/results/mnist_optionB/REPORT.md):

- 1000 MNIST training images drawn by data seed 42 + seed, and all 10000 test
  images, from mnist_data.load_mnist.
- ntk_lib.DeepBiasFreeMLP with three layers, standard parameterisation and the
  initial weights multiplied by the initial scale, trained uncentred with mean
  squared error on one-hot labels.
- Plain SGD with batch 200 and no momentum. The learning rate starts at
  eta_0 = 1 / lam_0, where lam_0 is the top Gauss-Newton eigenvalue of the
  loss at the run's own initial weights, and the curvature rule resets it to
  eta_0 lam_0 / lam_t every 10000 steps. The decay per step eta_kappa is fixed.
- The mixed probe of 128 training and 128 test images, and the closed-form
  kernel.

The initial scale plays the part of alpha in dataset_sweep.py. The two are
different knobs. Alpha scales the centred output, and the centred predictor
never grokked on MNIST in the exploration.

Event definitions. Memorisation is the first checkpoint with training accuracy
at least 0.99, as golden report section 3.5 defines it. The first checkpoint
with training accuracy 1 is recorded too, as dataset_sweep.py uses it. MNIST
never reaches the report's test levels of 0.90 and 0.95 reliably, so the test
levels here are 0.50 to 0.90, and the primary grokking event is test accuracy
0.80. This is a change of definition for MNIST only. A run is censored at its
step budget.

Run it from the mnist folder with the environment at the repository root.

    uv run python mnist_sweep.py check        # closed forms against autograd, logger leaves training unchanged
    uv run python mnist_sweep.py pilot        # timings under full load and the projected wall time
    uv run python mnist_sweep.py run          # the full sweep; it resumes where it stopped
    uv run python mnist_sweep.py status       # counts of finished runs
    uv run python mnist_sweep.py assemble     # the tables, the data dictionary and the manifest
    uv run python mnist_sweep.py validate     # the checks, also run by assemble

The options --scales, --widths, --decays, --seeds and --steps choose a part of
the grid, so the sweep can be run in batches, for example --widths 100 200.

Output goes to mnist/sweep/. The per-run files in sweep/runs/, the logs and
the long table checkpoints.parquet are ignored by git. runs.csv, runs.parquet,
the data dictionary and the manifest are small enough to commit.
"""

import os

# One thread per worker. Several workers that each start a BLAS thread pool
# overload the processor, which the exploration found with 12 workers.
for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import logging.handlers
import math
import multiprocessing as mp
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "repoduced-code"
for _p in (str(LIB), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = HERE / "sweep"
RUNS = OUT / "runs"
LOGS = OUT / "logs"
EXPLORATION = HERE / "exploration" / "results"

SCALES = [3.0, 5.0, 7.0, 9.0]
WIDTHS = [100, 200, 400, 800]
DECAYS = [0.0, 1e-6, 3e-6, 6e-6, 1e-5, 2e-5, 4e-5]
SEEDS = [0, 1, 2, 3, 4]
DATA_SEED_BASE = 42
STEPS = 500_000
N_TRAIN, BATCH_SIZE, DEPTH = 1000, 200, 3
LR_UPDATE_INTERVAL = 10_000
LINEAR_INTERVAL = 2000
N_LOG = 120
# Weights are saved at step 0, at the last step and at this many log-spaced
# checkpoints, for the seeds in WEIGHT_SEEDS. The other seeds save the last
# step only. A width 800 network holds about 1.3 million weights, so saving
# every seed at many checkpoints would take tens of gigabytes.
N_WEIGHT_LOG, WEIGHT_SEEDS = 6, (0,)
# Widths at or above this go to the pool of wide runs.
WIDE = 400
WIDE_WORKERS, NARROW_WORKERS = 3, 5

ACC_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
PRIMARY_LEVEL = 0.8
MEM_LEVEL = 0.99
LOSS_TAUS = (8e-2, 6e-2, 4e-2, 3e-2, 2e-2)
A_LEVELS = (0.45, 0.50)
EVENT_FIELDS = ("A_t", "S_c", "R_c", "gamma", "D", "weight_norm", "lr_factor", "S_sum", "R_sum", "D_sum", "A_sum",
                "S_W1", "S_W2", "S_W3")
TRAINING_FIELDS = ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm", "param_dist")

DESIGN_NOTE = ("The initial scale replaces alpha of dataset_sweep.py. The decay levels run from 0, where no run "
               "is expected to grok in the budget, through 3e-6 to 1e-5, where the exploration found strong "
               "grokking, to 2e-5, where test accuracy rose with too little delay, and 4e-5, expected to collapse "
               "the weights. Width 100 did not grok in the exploration and widths 200 to 800 did.")


# =====================================================================
# Design
# =====================================================================
def checkpoint_grid(steps):
    """Step 0, every LINEAR_INTERVAL steps, and N_LOG log-spaced steps. It holds the last step."""
    return sorted({0, steps, *range(0, steps + 1, LINEAR_INTERVAL), *log_steps(steps)})


def log_steps(steps):
    """The log-spaced part of the grid, rounded to integers."""
    return sorted(set(np.rint(np.logspace(0, np.log10(steps), N_LOG)).astype(int).tolist()))


def weight_steps(steps, grid, seed):
    """The checkpoints at which the weights are saved."""
    if seed not in WEIGHT_SEEDS:
        return [int(steps)]
    grid = np.asarray(grid)
    targets = np.logspace(0, np.log10(steps), N_WEIGHT_LOG)
    picked = {int(grid[np.argmin(np.abs(grid - t))]) for t in targets}
    return sorted(picked | {0, int(steps)})


def run_id(width, scale, eta_kappa, seed):
    return f"mnist_w{width}_s{scale:g}_ek{eta_kappa:g}_seed{seed}"


def run_spec(width, scale, eta_kappa, seed, steps):
    """The keyword arguments of ntk_lib.train_run for one run, without eta_0, which run_one sets from lam_0."""
    return dict(dataset="mnist", n_train=N_TRAIN, data_seed=DATA_SEED_BASE + seed, parameterisation="deep_standard",
                depth=DEPTH, hidden_dim=width, init_scale=scale, centred=False, alpha=1.0, optimizer="sgd",
                momentum=0.0, batch_size=BATCH_SIZE, eta_kappa=eta_kappa, lr_schedule="curvature",
                lr_update_interval=LR_UPDATE_INTERVAL, seed=seed, steps=steps, eval_interval=LINEAR_INTERVAL,
                kernel_save_interval=steps, probe="mixed", probe_size=256, kernel_method="closed_form",
                checkpoint_steps=checkpoint_grid(steps))


def build_tasks(scales, widths, decays, seeds, steps, out_dir, grid=None):
    tasks = []
    for width in widths:
        for scale in scales:
            for ek in decays:
                for seed in seeds:
                    spec = run_spec(width, scale, ek, seed, steps)
                    if grid is not None:
                        spec["checkpoint_steps"] = list(grid)
                    tasks.append(dict(run_id=run_id(width, scale, ek, seed), width=width, spec=spec,
                                      save_steps=weight_steps(steps, spec["checkpoint_steps"], seed),
                                      log_steps=log_steps(steps), out_dir=str(out_dir)))
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


def initial_curvature(spec):
    """lam_0, the top Gauss-Newton eigenvalue of the loss at the initial weights train_run will draw."""
    import torch
    import ntk_lib as L
    cfg = {**L.DEFAULTS, **spec}
    (X, y), _ = L.make_dataset(cfg)
    torch.manual_seed(cfg["seed"])
    model = L.build_model(cfg["parameterisation"], X.shape[1], cfg["hidden_dim"], y.shape[1], cfg["activation"],
                          cfg["init_scale"], depth=cfg["depth"])
    return L.gauss_newton_top_eigenvalue(model, X, alpha=cfg["alpha"])


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
    return dict(commit=git("rev-parse", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--", ".", str(LIB / "ntk_lib.py"))))


def versions():
    import torch
    import pandas
    return dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                pandas=pandas.__version__, platform=platform.platform(), machine=platform.machine(),
                host=platform.node(), torch_threads=torch.get_num_threads())


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
def closed_form_check(logger, weights, width, n_points=16):
    """Largest relative difference between the autograd and closed-form kernels at the saved steps.

    Both are computed in float64 from the saved float32 weights. In float32 a
    preactivation near zero can take a different sign in the two
    computations, which switches a ReLU unit and gives differences near 1e-3
    at large initial scale. The autograd kernel is summed one output at a time
    on the first n_points probe images, so a width 800 network needs a
    Jacobian of about 160 MB.
    """
    import torch
    from torch.func import functional_call, jacrev
    import ntk_lib as L
    import mnist_features as F
    x = logger.x_probe[:n_points].cpu().double()
    out = {}
    for step, Ws in weights.items():
        model = L.DeepBiasFreeMLP(x.shape[1], width, Ws[-1].shape[0], depth=DEPTH, scaling="standard").double()
        with torch.no_grad():
            for W, value in zip(model.weights(), Ws):
                W.copy_(torch.from_numpy(value))
        params = {k: v.detach() for k, v in model.named_parameters()}
        Ka = torch.zeros(len(x), len(x), dtype=torch.float64)
        for c in range(Ws[-1].shape[0]):
            J = jacrev(lambda p, m=model, c=c: functional_call(m, p, x)[:, c])(params)
            J = torch.cat([j.reshape(len(x), -1) for j in J.values()], dim=1)
            Ka += J @ J.T
            del J
        Kc = sum(F.kernel_terms(model.weights(), model.factors, x, np.eye(Ws[-1].shape[0])))
        out[str(step)] = (torch.linalg.norm(Ka - Kc) / torch.linalg.norm(Ka)).item()
    return out


def crossing(steps, values, level):
    """First checkpoint at which values reach level, or None."""
    idx = np.nonzero(np.asarray(values, float) >= level - 1e-12)[0]
    return int(steps[idx[0]]) if len(idx) else None


def run_events(run):
    """The memorisation and the primary grokking event, for the log."""
    h = run["history"]
    steps = np.asarray(h["step"])
    t_mem = crossing(steps, h["train_acc"], MEM_LEVEL)
    t_gen = crossing(steps, h["test_acc"], PRIMARY_LEVEL)
    return dict(t_mem=t_mem, t_gen=t_gen, censored=t_gen is None, best_test=float(max(h["test_acc"])))


def finished(task):
    """True if the task's meta file records a finished run of the same spec and feature version."""
    import mnist_features as F
    meta_path = Path(task["out_dir"]) / f"{task['run_id']}_meta.json"
    if not meta_path.exists():
        return False
    try:
        old = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (old.get("status") in ("ok", "diverged") and old.get("spec") == clean(task["spec"])
            and old.get("feature_version") == F.FEATURE_VERSION)


def log_start(log, task):
    s = task["spec"]
    log.info(f"{task['run_id']}: start, width {s['hidden_dim']}, scale {s['init_scale']:g}, eta_kappa "
             f"{s['eta_kappa']:g}, seed {s['seed']}, {s['steps']} steps", extra=event("start", run_id=task["run_id"]))


def run_one(task):
    """Train one run with the feature logger and write its files. Never raises."""
    import ntk_lib as L
    import mnist_features as F

    log = logging.getLogger("sweep")
    rid, out_dir = task["run_id"], Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if finished(task):
        return dict(run_id=rid, status="skipped", width=task["width"], seconds=0.0)

    s = task["spec"]
    log_start(log, task)
    t0 = time.time()
    error, run, logger, lam_0, config = None, None, None, None, None
    try:
        lam_0 = initial_curvature(s)
        config = dict(s, eta_0=1.0 / lam_0)
        cfg = L.normalise_checkpoints({**L.DEFAULTS, **config})
        logger = F.FeatureLogger(cfg, rid, save_steps=task["save_steps"], linear_interval=LINEAR_INTERVAL,
                                 log_steps=task["log_steps"])
        run = L.train_run(rid, verbose=False, on_checkpoint=logger, results_dir=out_dir, **config)
    except Exception:
        error = traceback.format_exc()
    return write_run(task, run, logger, lam_0, config, time.time() - t0, error)


def initial_curvatures(specs, device):
    """lam_0 of initial_curvature for several runs, from one vmapped power iteration on device.

    Each run's weights are drawn as initial_curvature draws them, and each
    run starts the power iteration from seed 0, as initial_curvature does.
    """
    import copy
    import torch
    from torch.func import stack_module_state
    import ntk_lib as L
    models, Xs, alphas = [], [], []
    for spec in specs:
        cfg = {**L.DEFAULTS, **spec}
        (X, y), _ = L.make_dataset(cfg)
        torch.manual_seed(cfg["seed"])
        models.append(L.build_model(cfg["parameterisation"], X.shape[1], cfg["hidden_dim"], y.shape[1],
                                    cfg["activation"], cfg["init_scale"], depth=cfg["depth"]))
        Xs.append(X)
        alphas.append(cfg["alpha"])
    params = {k: v.detach().to(device) for k, v in stack_module_state(models)[0].items()}
    X = torch.stack(Xs).to(device)
    alpha = torch.tensor(alphas, dtype=X.dtype, device=device)
    base = copy.deepcopy(models[0]).to("meta")
    return L.fused_gauss_newton_top_eigenvalue(base, params, X, alpha, seeds=[0] * len(specs))


def run_group(tasks, device=None):
    """Train tasks of one width as one fused model with ntk_lib.train_fused, and write each run's files.

    Each task gets the files and log lines run_one writes. Finished runs are
    skipped. An exception fails every run of the group. Never raises.
    """
    import torch
    import ntk_lib as L
    import mnist_features as F

    log = logging.getLogger("sweep")
    for t in tasks:
        Path(t["out_dir"]).mkdir(parents=True, exist_ok=True)
    results = [dict(run_id=t["run_id"], status="skipped", width=t["width"], seconds=0.0) for t in tasks
               if finished(t)]
    todo = [t for t in tasks if not finished(t)]
    if not todo:
        return results
    if len({t["out_dir"] for t in todo}) > 1:
        raise ValueError("the tasks of a fused group must share one output directory")
    for t in todo:
        log_start(log, t)
    device = torch.device(device) if device is not None else (
        torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu"))
    t0 = time.time()
    error, runs, loggers = None, None, None
    lams, configs = [None] * len(todo), [None] * len(todo)
    try:
        lams = initial_curvatures([t["spec"] for t in todo], device)
        configs = [dict(t["spec"], eta_0=1.0 / lam) for t, lam in zip(todo, lams)]
        loggers = [F.FeatureLogger(L.normalise_checkpoints({**L.DEFAULTS, **c}), t["run_id"],
                                   save_steps=t["save_steps"], linear_interval=LINEAR_INTERVAL,
                                   log_steps=t["log_steps"], device=device) for t, c in zip(todo, configs)]
        fused_logger = F.FusedFeatureLogger(loggers)
        runs = L.train_fused([t["run_id"] for t in todo], configs, results_dir=todo[0]["out_dir"], verbose=False,
                             device=device, on_fused_checkpoint=fused_logger)
        fused_logger.finish()
    except Exception:
        error = traceback.format_exc()
    seconds = time.time() - t0
    for i, t in enumerate(todo):
        results.append(write_run(t, runs[i] if runs else None, loggers[i] if loggers else None, lams[i],
                                 configs[i], seconds, error))
    return results


def write_run(task, run, logger, lam_0, config, seconds, error):
    """Write a trained run's meta file, feature table and weights, log its end, and return its summary."""
    import pandas as pd
    import mnist_features as F

    log = logging.getLogger("sweep")
    rid, out_dir = task["run_id"], Path(task["out_dir"])
    meta_path = out_dir / f"{rid}_meta.json"
    s = task["spec"]
    status = "failed" if run is None else ("diverged" if run["diverged"] else "ok")
    meta = dict(run_id=rid, status=status, error=error, spec=task["spec"], config=config, lam_0=lam_0,
                feature_version=F.FEATURE_VERSION, seconds=seconds, git=git_state(), versions=versions(),
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
                                **{layer: np.stack([weights[t][i] for t in steps])
                                   for i, layer in enumerate(F.LAYERS)})
            ends = {t: weights[t] for t in {steps[0], steps[-1]}}
            meta.update(n_checkpoints=len(logger.rows), last_step=logger.last_step,
                        callback_seconds=logger.callback_seconds,
                        closed_form_check=closed_form_check(logger, ends, s["hidden_dim"]),
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
        worst = max((v for v in meta.get("closed_form_check", {}).values() if v is not None), default=float("nan"))
        grok = ("censored" if e.get("censored") else f"test {PRIMARY_LEVEL:g} at {e.get('t_gen')}") \
            if e.get("t_mem") is not None else "never memorised"
        log.info(f"{rid}: {meta['status']} in {seconds:.0f} s, {meta.get('n_checkpoints')} checkpoints, "
                 f"memorised at {e.get('t_mem')}, {grok}, best test {e.get('best_test', float('nan')):.3f}, "
                 f"closed-form check {worst:.1e}",
                 extra=event("finish", run_id=rid, status=meta["status"], seconds=seconds, **e))
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
            if not group or n == 0:
                continue
            pool = ctx.Pool(min(n, len(group)), initializer=init_worker, initargs=(queue,), maxtasksperchild=20)
            pools.append(pool)
            for t in group:
                pool.apply_async(run_one, (t,), callback=done, error_callback=failed)
        total = sum(len(g) for g, n in ((wide, wide_workers), (narrow, narrow_workers)) if n > 0)
        if total < len(tasks):
            log.warning(f"{len(tasks) - total} tasks have no workers and were not started")
        last = 0.0
        while len(results) < total:
            time.sleep(5)
            if time.time() - last >= 300 or len(results) == total:
                last = time.time()
                counts = {s: sum(r["status"] == s for r in results) for s in ("ok", "diverged", "failed", "skipped")}
                trained = counts["ok"] + counts["diverged"] + counts["failed"]
                elapsed = time.time() - t_start
                rate = trained / elapsed * 3600 if trained else 0.0
                eta = (total - len(results)) / (trained / elapsed) if trained else float("nan")
                log.info(f"progress: {len(results)}/{total} done ({counts['ok']} ok, {counts['diverged']} diverged, "
                         f"{counts['failed']} failed, {counts['skipped']} skipped), {rate:.1f} runs per hour, "
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
def cmd_check(args, log):
    """The closed-form kernels against autograd, and the logger's effect on training."""
    import torch
    from torch.func import functional_call, jacrev
    import ntk_lib as L
    import mnist_features as F

    ok = True
    (Xtr, ytr), (Xte, yte) = L.make_dataset({**L.DEFAULTS, "dataset": "mnist", "n_train": N_TRAIN})
    x, _, _ = L.select_probe(Xtr, ytr, Xte, yte, "mixed", 256)
    x = x[torch.arange(0, 256, 16)]
    worst = 0.0
    for width in WIDTHS:
        for moved in (False, True):
            torch.manual_seed(0)
            model = L.build_model("deep_standard", 784, width, 10, init_scale=7.0, depth=DEPTH).double()
            if moved:
                with torch.no_grad():
                    for q in model.parameters():
                        q.mul_(0.4).add_(0.05 * torch.randn_like(q))
            xd = x.double()
            params = {k: v.detach() for k, v in model.named_parameters()}
            J = jacrev(lambda p, m=model, xd=xd: functional_call(m, p, xd))(params)
            J = torch.cat([j.reshape(len(xd), 10, -1) for j in J.values()], dim=2)   # n by C by P
            for name, U in (("trace", np.eye(10)), ("first logit", np.eye(10)[:, :1]),
                            ("sum over sqrt(C)", np.ones((10, 1)) / math.sqrt(10))):
                JU = torch.einsum("ncp,cm->nmp", J, torch.as_tensor(U))
                Ka = torch.einsum("nmp,kmp->nk", JU, JU)
                Kc = sum(F.kernel_terms(model.weights(), model.factors, xd, U))
                d = (torch.linalg.norm(Ka - Kc) / torch.linalg.norm(Ka)).item()
                worst = max(worst, d)
                log.info(f"check: width {width:4d} {'moved' if moved else 'init '} {name:>16} kernel: {d:.1e}")
    ok &= worst < 1e-10
    log.info(f"check: largest relative difference {worst:.1e} ({'pass' if worst < 1e-10 else 'FAIL'}, limit 1e-10)")

    # The logger must leave training unchanged, the schedule included.
    scratch = OUT / "check"
    steps = 3000
    spec = run_spec(100, 7.0, 1e-5, 1, steps)
    spec.update(lr_update_interval=1000, checkpoint_steps=checkpoint_grid(steps))
    config = dict(spec, eta_0=1.0 / initial_curvature(spec))
    plain = L.train_run("plain", verbose=False, results_dir=scratch, **config)
    full = L.normalise_checkpoints({**L.DEFAULTS, **config})
    logger = F.FeatureLogger(full, "logged", save_steps=[0], log_steps=log_steps(steps))
    logged = L.train_run("logged", verbose=False, on_checkpoint=logger, results_dir=scratch, **config)
    keys = [k for k in plain["history"] if k in logged["history"]]
    diff = max(float(np.nanmax(np.abs(np.asarray(plain["history"][k], float) -
                                      np.asarray(logged["history"][k], float)))) for k in keys)
    ok &= diff == 0.0
    log.info(f"check: history with and without the logger differs by {diff} over {len(plain['history']['step'])} "
             f"checkpoints and {len(keys)} keys ({'pass' if diff == 0 else 'FAIL'})")
    ident = F.identity_checks(logger.check_arrays())
    for k, v in ident.items():
        log.info(f"check: identity {k}: {v:.1e}")
    ident_ok = all(v < 1e-4 for v in ident.values() if math.isfinite(v))
    ok &= ident_ok
    log.info(f"check: identities {'pass' if ident_ok else 'FAIL'} (limit 1e-4, float32 rounding of the history)")
    # The fused trainer must give each run the history train_run gives it. In float32 the two drift apart once
    # the curvature rule raises the learning rate, because SGD amplifies their different rounding. In float64
    # the same amplification stays below the float64 tolerances of torch.testing.assert_close, while an error
    # in the fused step would not.
    specs = []
    for seed in (1, 2):
        spec = run_spec(100, 7.0, 1e-5, seed, steps)
        spec.update(lr_update_interval=1000, checkpoint_steps=checkpoint_grid(steps), dtype="float64")
        specs.append(dict(spec, eta_0=1.0 / initial_curvature(spec)))
    serial = [L.train_run(f"serial{i}", verbose=False, results_dir=scratch, **c) for i, c in enumerate(specs)]
    fused = L.train_fused([f"fused{i}" for i in range(len(specs))], specs, results_dir=scratch, verbose=False,
                          device="cpu")
    fused_ok, worst_fused = True, 0.0
    for a, b in zip(fused, serial):
        for k in b["history"]:
            x = torch.tensor(np.asarray(a["history"][k], float), dtype=torch.float64)
            y = torch.tensor(np.asarray(b["history"][k], float), dtype=torch.float64)
            if x.shape != y.shape:
                fused_ok = False
                log.info(f"check: fused history {k} has {len(x)} checkpoints, train_run {len(y)}")
                continue
            worst_fused = max(worst_fused, float(torch.nan_to_num((x - y).abs()).max()))
            try:
                torch.testing.assert_close(x, y, equal_nan=True)
            except AssertionError as error:
                fused_ok = False
                log.info(f"check: fused history {k}: {str(error).splitlines()[0]}")
    ok &= fused_ok
    log.info(f"check: fused against train_run, largest difference {worst_fused:.1e} over {len(specs)} runs "
             f"({'pass' if fused_ok else 'FAIL'}, torch.testing.assert_close float64 defaults)")

    # The fused feature logger must give each run the rows FeatureLogger gives it, also in float64.
    def logger_for(i, c, prefix):
        return F.FeatureLogger(L.normalise_checkpoints({**L.DEFAULTS, **c}), f"{prefix}{i}", save_steps=[0, steps],
                               log_steps=log_steps(steps), dtype=torch.float64)

    serial_loggers = [logger_for(i, c, "serial") for i, c in enumerate(specs)]
    for i, (c, lg) in enumerate(zip(specs, serial_loggers)):
        L.train_run(f"serial{i}", verbose=False, on_checkpoint=lg, results_dir=scratch, **c)
    fused_loggers = [logger_for(i, c, "fused") for i, c in enumerate(specs)]
    group = F.FusedFeatureLogger(fused_loggers)
    L.train_fused([f"fused{i}" for i in range(len(specs))], specs, results_dir=scratch, verbose=False, device="cpu",
                  on_fused_checkpoint=group)
    group.finish()
    rows_ok, worst_rows = True, 0.0
    skip = {"run_id", "wall_seconds"}
    for a, b in zip(fused_loggers, serial_loggers):
        if len(a.rows) != len(b.rows):
            rows_ok = False
            log.info(f"check: fused logger has {len(a.rows)} rows, FeatureLogger {len(b.rows)}")
            continue
        for k in F.COLUMN_NAMES:
            if k in skip:
                continue
            x = torch.tensor([float(r[k]) for r in a.rows], dtype=torch.float64)
            y = torch.tensor([float(r[k]) for r in b.rows], dtype=torch.float64)
            worst_rows = max(worst_rows, float(torch.nan_to_num((x - y).abs()).max()))
            try:
                torch.testing.assert_close(x, y, equal_nan=True)
            except AssertionError as error:
                rows_ok = False
                log.info(f"check: fused logger column {k}: {str(error).splitlines()[0]}")
    ok &= rows_ok
    log.info(f"check: fused logger against FeatureLogger, largest difference {worst_rows:.1e} over "
             f"{len(F.COLUMN_NAMES) - len(skip)} columns ({'pass' if rows_ok else 'FAIL'}, "
             "torch.testing.assert_close float64 defaults)")
    log.info("check: all passed" if ok else "check: SOME CHECKS FAILED")
    return 0 if ok else 1


def run_fused(tasks, log):
    """Run the tasks as one fused group for each width, widest first, and log each group's end."""
    results, t_start = [], time.time()
    for width in sorted({t["width"] for t in tasks}, reverse=True):
        group = [t for t in tasks if t["width"] == width]
        t0 = time.time()
        done = run_group(group)
        results += done
        counts = {s: sum(r["status"] == s for r in done) for s in ("ok", "diverged", "failed", "skipped")}
        log.info(f"fused: width {width} finished in {(time.time() - t0) / 3600:.2f} h, {counts}, elapsed "
                 f"{(time.time() - t_start) / 3600:.2f} h", extra=event("fused_group", width=width, **counts))
    return results


def pilot_fused(args, log):
    """Short fused runs of the whole grid at every width, and the projected wall time of the fused sweep.

    Each width trains all its runs of the chosen grid as one fused model for
    3000 steps, with the checkpoints of cmd_pilot. The wall times of the
    first run of the group give the seconds of one fused step and of one
    checkpoint of all the runs, by the same two equations as cmd_pilot.
    """
    import pandas as pd
    out_dir = OUT / "pilot_fused"
    steps = 3000
    grid = list(range(0, 1001, 50)) + [steps]
    per_width = {}
    for width in args.widths:
        tasks = build_tasks(args.scales, [width], args.decays, args.seeds, steps, out_dir, grid)
        for t in tasks:
            t.update(run_id=f"pilot_{t['run_id']}", save_steps=[steps], log_steps=[])
            meta = out_dir / f"{t['run_id']}_meta.json"
            if meta.exists():
                meta.unlink()
        t0 = time.time()
        results = run_group(tasks)
        failed = [r["run_id"] for r in results if r["status"] == "failed"]
        if failed:
            log.error(f"pilot fused: width {width} failed for {len(failed)} runs, see sweep.log")
            return 1
        frame = pd.read_parquet(out_dir / f"{tasks[0]['run_id']}.parquet")
        wall = dict(zip(frame["step"], frame["wall_seconds"]))
        # From step 0 to 1000 there are 1000 steps and 20 checkpoints; from 1000 to 3000, 2000 steps and 1.
        A = np.array([[1000.0, 20.0], [2000.0, 1.0]])
        b = np.array([wall[1000] - wall[0], wall[steps] - wall[1000]])
        per_width[width] = (np.linalg.solve(A, b), len(tasks), time.time() - t0)
    n_ckpt = len(checkpoint_grid(args.steps))
    total = 0.0
    for width in sorted(per_width):
        (step_s, ckpt_s), n_runs, wall_s = per_width[width]
        per_group = args.steps * step_s + n_ckpt * ckpt_s
        total += per_group
        log.info(f"pilot fused: width {width:4d}: {n_runs} runs as one model, {step_s * 1000:.3f} ms per fused "
                 f"step, {ckpt_s * 1000:.0f} ms per checkpoint of all runs, {per_group / 3600:.2f} h for the width, "
                 f"pilot wall {wall_s:.0f} s")
    log.info(f"pilot fused: projected wall time {total / 3600:.2f} h with the widths one after another on one "
             f"device, {n_ckpt} checkpoints of {args.steps} steps. Learning rate updates are counted only if they "
             "fall in the pilot, so the projection is a little low.")
    return 0


def cmd_pilot(args, log):
    """Short runs at every width under the real concurrency, and the projected wall time of the sweep."""
    if args.fused:
        return pilot_fused(args, log)
    out_dir = OUT / "pilot"
    steps = 3000
    grid = list(range(0, 1001, 50)) + [steps]
    def runs_per_width(width):
        """Enough runs to fill the width's pool, shared evenly between the widths in that pool."""
        same_pool = [w for w in args.widths if (w >= WIDE) == (width >= WIDE)]
        workers = args.wide_workers if width >= WIDE else args.narrow_workers
        return max(1, workers // len(same_pool))

    tasks = []
    for width in args.widths:
        for k in range(runs_per_width(width)):
            spec = run_spec(width, 7.0, 1e-5, k, steps)
            spec["checkpoint_steps"] = grid
            tasks.append(dict(run_id=f"pilot_{run_id(width, 7.0, 1e-5, k)}", width=width, spec=spec,
                              save_steps=[steps], log_steps=[], out_dir=str(out_dir)))
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
    n_lr = args.steps // LR_UPDATE_INTERVAL
    runs_per_width = len(args.scales) * len(args.decays) * len(args.seeds)
    wide_total = narrow_total = 0.0
    log.info(f"pilot: {n_ckpt} checkpoints per run of {args.steps} steps, {runs_per_width} runs per width. The "
             f"{n_lr} learning rate updates per run are counted in the per-step time only if they fall in the "
             "pilot, so the projection is a little low.")
    for width in sorted(per_width):
        step_s, ckpt_s = np.mean(per_width[width], axis=0)
        per_run = args.steps * step_s + n_ckpt * ckpt_s
        if width >= WIDE:
            wide_total += per_run * runs_per_width
        else:
            narrow_total += per_run * runs_per_width
        log.info(f"pilot: width {width:4d}: {step_s * 1000:.3f} ms per step, {ckpt_s * 1000:.0f} ms per checkpoint, "
                 f"{per_run / 60:.1f} min per run")
    wide_wall = wide_total / max(args.wide_workers, 1)
    narrow_wall = narrow_total / max(args.narrow_workers, 1)
    log.info(f"pilot: projected wall time {max(wide_wall, narrow_wall) / 3600:.1f} h (wide pool {wide_wall / 3600:.1f} "
             f"h on {args.wide_workers} workers, narrow pool {narrow_wall / 3600:.1f} h on {args.narrow_workers} "
             "workers). The rates were measured with both pools full, so the pool that finishes first frees the "
             "processor for the other.")
    return 0


def cmd_run(args, log):
    tasks = build_tasks(args.scales, args.widths, args.decays, args.seeds, args.steps, RUNS)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "commands.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, git=git_state())) + "\n")
    log.info(f"run: {len(tasks)} runs, {args.wide_workers} wide and {args.narrow_workers} narrow workers, "
             f"{len(checkpoint_grid(args.steps))} checkpoints per run, output in {RUNS}",
             extra=event("sweep_start", n_runs=len(tasks)))
    if args.fused:
        results = run_fused(tasks, log)
    else:
        results = run_pools(tasks, args.wide_workers, args.narrow_workers, log)
    counts = {s: sum(r["status"] == s for r in results) for s in ("ok", "diverged", "failed", "skipped")}
    log.info(f"run: finished, {counts}", extra=event("sweep_finish", **counts))
    return 0 if counts["failed"] == 0 else 1


def cmd_status(args, log):
    metas = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*_meta.json"))]
    total = len(build_tasks(args.scales, args.widths, args.decays, args.seeds, args.steps, RUNS))
    counts = {}
    for m in metas:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    print(f"{len(metas)} runs have finished, of {total} in the chosen grid: {counts}")
    by_width = {}
    for m in metas:
        by_width.setdefault(m["spec"]["hidden_dim"], []).append(m["seconds"])
    for width in sorted(by_width):
        s = by_width[width]
        print(f"  width {width:4d}: {len(s):4d} runs, mean {np.mean(s) / 60:.0f} min per run")
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


rcol("run_id", "The name of the run, run_id.")
rcol("init_scale", "The initial scale s.", "Liu, Michaud and Tegmark (2023), Omnigrok, arXiv:2210.01117, Section 4")
rcol("width", "The width of both hidden layers.")
rcol("eta_kappa", "The weight decay per step, eta lambda.", "docs/report.tex, The laziness and weight-decay knobs")
rcol("seed", "The model seed.")
rcol("data_seed", "The seed of the draw of the 1000 training images, 42 + seed.")
rcol("n_train", "The number of training images.")
rcol("batch_size", "The minibatch size.")
rcol("lam_0", "The top Gauss-Newton eigenvalue of the loss at the initial weights.",
     "ntk_lib.gauss_newton_top_eigenvalue")
rcol("eta_0", "The initial learning rate, 1 / lam_0.")
rcol("status", "ok, diverged (the training loss stopped being finite) or failed (an exception).")
rcol("error", "The traceback of a failed run, empty otherwise.")
rcol("diverged_step", "The checkpoint at which the training loss was first not finite.")
rcol("steps", "The step budget.")
rcol("last_step", "The last recorded checkpoint.")
rcol("n_checkpoints", "The number of recorded checkpoints.")
rcol("seconds", "Wall-clock seconds for the run, logger included.")
rcol("callback_seconds", "Wall-clock seconds spent in the feature logger.")
rcol("feature_version", "mnist_features.FEATURE_VERSION when the run was made.")
rcol("git_commit", "The commit of the repository when the run was made.")
rcol("git_dirty", "True if mnist/ or ntk_lib.py had uncommitted changes when the run was made.")
rcol("closed_form_check", "Largest relative difference between the autograd and closed-form kernels on 32 probe "
     "images at the first and last checkpoints.", "ntk_lib.entk_closed_form_deep")
rcol("init_weight_norm", "Weight norm at step 0.", "ntk_lib.train_run")
rcol("final_weight_norm", "Weight norm at the last checkpoint.")
rcol("weight_norm_growth", "final_weight_norm / init_weight_norm.")
rcol("final_lr_factor", "The learning rate at the last checkpoint over eta_0.")
for name in ("A_t", "S_c", "R_c", "S_sum", "R_sum", "D_sum", "A_sum", "train_acc", "test_acc", "train_loss",
             "test_loss"):
    rcol(f"{name}_final", f"{name} at the last checkpoint.")
rcol("A_0", "A_t at step 0.")
rcol("A_peak", "Largest A_t over the run.")
rcol("A_peak_step", "The step of A_peak.")
rcol("A_min", "Smallest A_t over the run.")
rcol("A_min_step", "The step of A_min.")
rcol("best_test_acc", "Largest test accuracy over the run.")
rcol("best_test_step", "The step of best_test_acc.")
rcol("memorised", f"True if the training accuracy reached {MEM_LEVEL} at some checkpoint. A run that never "
     "memorises has no grokking time, which is different from a censored one.", "docs/report_golden.pdf, section 3.5")
rcol("t_mem", f"First checkpoint at which the training accuracy is at least {MEM_LEVEL}.",
     "docs/report_golden.pdf, section 3.5")
rcol("t_mem_prev", "The checkpoint before t_mem, so the event lies in (t_mem_prev, t_mem].")
rcol("t_mem100", "First checkpoint at which the training accuracy is 1, the memorisation of dataset_sweep.py.",
     "ntk_lib.accuracy_crossings")
rcol("test_acc_at_mem", "Test accuracy at t_mem.")
for level in ACC_LEVELS:
    tag = acc_tag(level)
    rcol(f"t_gen_acc{tag}", f"First checkpoint at which the test accuracy reaches {level:g}, counted from step 0.")
    rcol(f"t_gen_acc{tag}_prev", f"The checkpoint before t_gen_acc{tag}.")
    rcol(f"t_grok_acc{tag}", f"Grokking time at test level {level:g}: t_gen_acc{tag} - t_mem, or last_step - t_mem "
         "if censored. Empty if the run never memorised. Negative if the level came before memorisation.",
         "dataset_sweep.py, the same column")
    rcol(f"censored_acc{tag}", f"True if the test accuracy never reached {level:g}, or the run never memorised.")
for tau in LOSS_TAUS:
    tag = loss_tag(tau)
    rcol(f"t_train_loss{tag}", f"First checkpoint at which the training loss is below {tau:g}.",
         "ntk_lib.grokking_time")
    rcol(f"t_test_loss{tag}", f"First checkpoint at which the test loss is below {tau:g}.", "ntk_lib.grokking_time")
    rcol(f"t_grok_loss{tag}", f"Grokking time under the loss definition with both thresholds {tau:g}.",
         "ntk_lib.grokking_time; docs/report.tex, Defining t_grok")
    rcol(f"censored_loss{tag}", f"True if the test loss never fell below {tau:g}, or the training loss never did.",
         "ntk_lib.grokking_time")
for when, word in (("mem", "t_mem"), ("gen", f"t_gen_acc{acc_tag(PRIMARY_LEVEL)}")):
    for name in EVENT_FIELDS:
        rcol(f"{name}_at_{when}", f"{name} at {word}. Empty if the event did not happen.")
for level in A_LEVELS:
    rcol(f"t_A{a_tag(level)}", f"First checkpoint at which A_t reaches {level:g}. Empty if it never does.")
rcol("verdict", "strong, weak or none, by the criterion of exploration/PLAN_mnist_closed_form.md section 3 with "
     f"memorisation at {MEM_LEVEL}. Strong: test accuracy at t_mem at most 0.30, best test accuracy at least 0.80, "
     "and t_gen_acc50 / t_mem at least 3. Weak: not strong, a gain after t_mem of at least 0.25, and the step at "
     "half that gain at least 3 t_mem.", "exploration/PLAN_mnist_closed_form.md, section 3")


def previous_step(steps, t):
    if t is None:
        return None
    i = int(np.searchsorted(steps, t))
    return int(steps[i - 1]) if i > 0 else None


def verdict(steps, test_acc, t_mem):
    if t_mem is None:
        return "none"
    test_acc = np.asarray(test_acc, float)
    at_mem = float(test_acc[steps == t_mem][0])
    best = float(test_acc.max())
    t50 = crossing(steps, test_acc, 0.5)
    if at_mem <= 0.30 and best >= 0.80 and t50 is not None and t50 > t_mem and t50 / t_mem >= 3:
        return "strong"
    gain = best - at_mem
    t_half = crossing(steps, test_acc, at_mem + gain / 2)
    if gain >= 0.25 and t_half is not None and t_half > t_mem and t_half / t_mem >= 3:
        return "weak"
    return "none"


def run_row(meta, frame, out_dir):
    """One row of the runs table, from a run's meta file, its feature rows and its train_run JSON."""
    import ntk_lib as L
    s = meta["spec"]
    c = meta.get("config") or {}
    row = dict(run_id=meta["run_id"], init_scale=s["init_scale"], width=s["hidden_dim"], eta_kappa=s["eta_kappa"],
               seed=s["seed"], data_seed=s["data_seed"], n_train=s["n_train"], batch_size=s["batch_size"],
               lam_0=meta.get("lam_0"), eta_0=c.get("eta_0"), status=meta["status"], error=meta.get("error") or "",
               diverged_step=meta.get("diverged_step"), steps=s["steps"], last_step=meta.get("last_step"),
               n_checkpoints=meta.get("n_checkpoints"), seconds=meta["seconds"],
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
               weight_norm_growth=h["weight_norm"][-1] / run["init_weight_norm"],
               final_lr_factor=float(f["lr_factor"].iloc[-1]))
    for name in ("A_t", "S_c", "R_c", "S_sum", "R_sum", "D_sum", "A_sum", "train_acc", "test_acc", "train_loss",
                 "test_loss"):
        row[f"{name}_final"] = float(f[name].iloc[-1])
    row.update(A_0=float(f["A_t"].iloc[0]), A_peak=float(f["A_t"].max()), A_peak_step=int(f["A_t"].idxmax()),
               A_min=float(f["A_t"].min()), A_min_step=int(f["A_t"].idxmin()),
               best_test_acc=float(f["test_acc"].max()), best_test_step=int(f["test_acc"].idxmax()))
    t_mem = crossing(steps, h["train_acc"], MEM_LEVEL)
    row.update(memorised=t_mem is not None, t_mem=t_mem, t_mem_prev=previous_step(steps, t_mem),
               t_mem100=crossing(steps, h["train_acc"], 1.0),
               test_acc_at_mem=float(f.loc[t_mem, "test_acc"]) if t_mem is not None else None)
    last = int(steps[-1])
    for level in ACC_LEVELS:
        tag = acc_tag(level)
        t_gen = crossing(steps, h["test_acc"], level)
        row[f"t_gen_acc{tag}"] = t_gen
        row[f"t_gen_acc{tag}_prev"] = previous_step(steps, t_gen)
        if t_mem is None:
            row[f"t_grok_acc{tag}"], row[f"censored_acc{tag}"] = None, True
        elif t_gen is None:
            row[f"t_grok_acc{tag}"], row[f"censored_acc{tag}"] = last - t_mem, True
        else:
            row[f"t_grok_acc{tag}"], row[f"censored_acc{tag}"] = t_gen - t_mem, False
    for tau in LOSS_TAUS:
        g = L.grokking_time(run, tau, tau)
        tag = loss_tag(tau)
        row.update({f"t_train_loss{tag}": g["t_train"], f"t_test_loss{tag}": g["t_test"],
                    f"t_grok_loss{tag}": g["t_grok"], f"censored_loss{tag}": g["censored"]})
    t_primary = row[f"t_gen_acc{acc_tag(PRIMARY_LEVEL)}"]
    for when, t in (("mem", t_mem), ("gen", t_primary)):
        for name in EVENT_FIELDS:
            row[f"{name}_at_{when}"] = float(f.loc[t, name]) if t is not None else None
    for level in A_LEVELS:
        row[f"t_A{a_tag(level)}"] = crossing(f.index.values, f["A_t"].values, level)
    row["verdict"] = verdict(steps, h["test_acc"], t_mem)
    return row


def write_dictionary(out):
    import pandas as pd
    import mnist_features as F
    entries = [dict(table="checkpoints", **c) for c in F.COLUMNS] + [dict(table="runs", **c) for c in RUN_COLUMNS]
    frame = pd.DataFrame(entries, columns=["table", "name", "group", "meaning", "formula", "source", "units"])
    frame.to_csv(out / "data_dictionary.csv", index=False)
    lines = ["# Data dictionary of the MNIST sweep", "",
             "This file is written by `mnist_sweep.py assemble` from `mnist_features.COLUMNS` and "
             "`mnist_sweep.RUN_COLUMNS`. Edit those, not this file.", "",
             "`checkpoints.parquet` has one row per run and checkpoint. `runs.parquet` and `runs.csv` have one row "
             "per run. The two tables join on `run_id`. The weights are in `runs/<run_id>_weights.npz`, with arrays "
             "`steps`, `W1` of shape (T, N, 784), `W2` of shape (T, N, N) and `W3` of shape (T, 10, N). Seed 0 saves "
             f"about {N_WEIGHT_LOG + 2} checkpoints and the other seeds the last one. "
             "`mnist_features.kernel_terms` rebuilds any kernel from them.", "",
             "Notation. H = I - (1/n) 1 1^T. Kc = H K H is the centred kernel and k = Kc / ||Kc||_F the unit "
             "centred kernel. Y is the one-hot label matrix and G = H Y Y^T H. <A, B> is the Frobenius inner "
             "product. The probe is the mixed probe of 128 training and 128 test images.", ""]
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
    import mnist_features as F

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
    ckpt = pd.concat(frames, ignore_index=True).sort_values(["width", "init_scale", "eta_kappa", "seed", "step"])
    if list(ckpt.columns) != F.COLUMN_NAMES:
        raise AssertionError(f"checkpoint columns do not match mnist_features.COLUMNS: "
                             f"{sorted(set(ckpt.columns) ^ set(F.COLUMN_NAMES))}")
    ckpt.to_parquet(OUT / "checkpoints.parquet", index=False)
    # The report kernel table in the layout of model_fitting/report_kernel.py, which Zain's notebooks read.
    # That script rebuilds the statistics from saved weights at 48 steps. Here they are at every checkpoint.
    report = ckpt[["run_id", "step", "S_sum", "R_sum", "D_sum", "A_sum"]].rename(
        columns=dict(S_sum="S_t", R_sum="R_t", D_sum="D_t", A_sum="A_t"))
    report.to_parquet(OUT / "report_kernel.parquet", index=False)
    runs = pd.DataFrame(rows, columns=[c["name"] for c in RUN_COLUMNS])
    extra = set(pd.DataFrame(rows).columns) - set(runs.columns)
    if extra:
        raise AssertionError(f"run columns missing from RUN_COLUMNS: {sorted(extra)}")
    runs = runs.sort_values(["width", "init_scale", "eta_kappa", "seed"])
    runs.to_parquet(OUT / "runs.parquet", index=False)
    runs.to_csv(OUT / "runs.csv", index=False)
    write_dictionary(OUT)

    tag = acc_tag(PRIMARY_LEVEL)
    ok = runs[runs["status"] != "failed"]
    per_cell = []
    for (w, sc, ek), g in ok.groupby(["width", "init_scale", "eta_kappa"]):
        mem = g["memorised"].astype(bool)
        cens = g[f"censored_acc{tag}"].astype(bool)
        per_cell.append(dict(width=int(w), init_scale=float(sc), eta_kappa=float(ek), runs=len(g),
                             never_memorised=int((~mem).sum()), reached_acc80=int((mem & ~cens).sum()),
                             censored_acc80=int((mem & cens).sum()), strong=int((g["verdict"] == "strong").sum()),
                             weak=int((g["verdict"] == "weak").sum())))
    design = dict(init_scales=sorted(runs["init_scale"].unique().tolist()),
                  widths=sorted(runs["width"].unique().tolist()),
                  eta_kappas=sorted(runs["eta_kappa"].unique().tolist()), seeds=sorted(runs["seed"].unique().tolist()),
                  steps=sorted(runs["steps"].unique().tolist()))
    manifest = dict(
        written=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, git=git_state(), versions=versions(),
        design=design,
        fixed=dict(dataset="mnist", n_train=N_TRAIN, data_seed=f"{DATA_SEED_BASE} + seed", test_images=10000,
                   parameterisation="deep_standard", depth=DEPTH, biases=False, centred=False, loss="mse",
                   optimizer="sgd", momentum=0.0, batch_size=BATCH_SIZE, eta_0="1 / lam_0",
                   lr_schedule="curvature", lr_update_interval=LR_UPDATE_INTERVAL, probe="mixed", probe_size=256,
                   kernel_method="closed_form", linear_interval=LINEAR_INTERVAL, n_log=N_LOG),
        feature_version=F.FEATURE_VERSION, krr_ridge=F.KRR_RIDGE, sharpness_iters=F.SHARPNESS_ITERS,
        report_kernel=dict(test_images=F.REPORT_SIZE, seed=F.REPORT_SEED),
        events=dict(memorisation=MEM_LEVEL, primary_test_level=PRIMARY_LEVEL, acc_levels=ACC_LEVELS,
                    loss_taus=LOSS_TAUS, a_levels=A_LEVELS,
                    note="MNIST does not reach the report's test levels 0.90 and 0.95 reliably, so the primary "
                         "grokking event is test accuracy 0.80. This applies to MNIST only."),
        counts=dict(runs=len(runs), rows=len(ckpt), columns=len(ckpt.columns),
                    status=runs["status"].value_counts().to_dict(), verdict=runs["verdict"].value_counts().to_dict()),
        total_seconds=float(runs["seconds"].sum()), per_cell=per_cell, design_note=DESIGN_NOTE,
        files=dict(checkpoints="checkpoints.parquet", runs=["runs.parquet", "runs.csv"],
                   report_kernel="report_kernel.parquet: run_id, step, S_t, R_t, D_t, A_t of golden report "
                                 "Equation 6, copied from S_sum, R_sum, D_sum and A_sum",
                   dictionary=["data_dictionary.csv", "data_dictionary.md"], weights="runs/<run_id>_weights.npz",
                   run_json="runs/<run_id>.json", logs="logs/sweep.log, logs/events.jsonl"))
    write_atomic(OUT / "manifest.json", json.dumps(clean(manifest), indent=1))
    log.info(f"assemble: {len(runs)} runs, {len(ckpt)} checkpoint rows x {len(ckpt.columns)} columns, "
             f"{(OUT / 'checkpoints.parquet').stat().st_size / 1e6:.0f} MB; status {manifest['counts']['status']}; "
             f"verdicts {manifest['counts']['verdict']}")
    return cmd_validate(args, log)


# =====================================================================
# Validation
# =====================================================================
def exploration_runs():
    """The saved exploration runs, keyed by (width, scale, eta_kappa, seed, data_seed), that match the sweep setup."""
    found = {}
    for path in EXPLORATION.glob("*/*.json"):
        if path.name.startswith("summary") or path.name.endswith("analysis.json"):
            continue
        try:
            run = json.loads(path.read_text())
            c = run["config"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not (c.get("dataset") == "mnist" and c.get("parameterisation") == "deep_standard"
                and c.get("depth") == DEPTH and c.get("lr_schedule") == "curvature" and c.get("batch_size") == BATCH_SIZE
                and c.get("n_train") == N_TRAIN and not c.get("centred") and c.get("momentum") == 0.0
                and c.get("lr_update_interval") == LR_UPDATE_INTERVAL):
            continue
        found[(c["hidden_dim"], c["init_scale"], c["eta_kappa"], c["seed"], c["data_seed"])] = path
    return found


def cmd_validate(args, log):
    metas = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*_meta.json"))]
    good = [m for m in metas if m["status"] != "failed"]
    out = {}

    cf = [max(v for v in m["closed_form_check"].values() if v is not None) for m in good
          if m.get("closed_form_check") and any(v is not None for v in m["closed_form_check"].values())]
    out["closed_form"] = dict(runs=len(cf), largest=max(cf, default=None), limit=1e-4,
                              passed=bool(cf) and max(cf) < 1e-4)

    names = sorted({k for m in good for k in m.get("identity_checks", {})})
    out["identities"] = {k: max((m["identity_checks"][k] for m in good
                                 if m.get("identity_checks", {}).get(k) is not None), default=None) for k in names}

    # Reproduction against the exploration runs with the same setup. The training does not depend on the
    # checkpoints, so the training fields must agree exactly at the checkpoints the two runs share.
    saved = exploration_runs()
    compared, exact, mismatches = 0, 0, []
    for m in good:
        s = m["spec"]
        key = (s["hidden_dim"], s["init_scale"], s["eta_kappa"], s["seed"], s["data_seed"])
        mine_path = RUNS / f"{m['run_id']}.json"
        if key not in saved or not mine_path.exists():
            continue
        old, mine = json.loads(saved[key].read_text())["history"], json.loads(mine_path.read_text())["history"]
        index = {t: i for i, t in enumerate(mine["step"])}
        shared = [(index[t], j) for j, t in enumerate(old["step"]) if t in index]
        if not shared:
            continue
        compared += 1
        largest = max(abs(mine[k][i] - old[k][j]) for i, j in shared for k in TRAINING_FIELDS)
        if largest == 0:
            exact += 1
        else:
            mismatches.append(dict(run_id=m["run_id"], against=str(saved[key].relative_to(HERE)), largest=largest))
    out["reproduction"] = dict(runs_compared=compared, exact_in_training_fields=exact, mismatches=mismatches,
                               note="Exploration runs with the same width, scale, decay, seed and data seed must "
                                    "agree exactly in the training fields at the shared checkpoints, if they were "
                                    "trained with the same number of torch threads. The sweep uses one. The runs "
                                    "of exploration/results/mnist_grid used two, which changes the order of "
                                    "floating point sums, so they differ at rounding level and drift apart.")

    OUT.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT / "validation.json", json.dumps(clean(out), indent=1))
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["validation"] = clean(out)
        write_atomic(manifest_path, json.dumps(manifest, indent=1))
    log.info(f"validate: closed form largest {out['closed_form']['largest']} over {len(cf)} runs; "
             f"reproduction {exact}/{compared} runs exact in training fields")
    for k, v in out["identities"].items():
        log.info(f"validate: {k:>45}: largest {v}")
    return 0


# =====================================================================
# Entry point
# =====================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["check", "pilot", "run", "status", "assemble", "validate"])
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument("--decays", type=float, nargs="+", default=DECAYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--wide-workers", type=int, default=WIDE_WORKERS)
    parser.add_argument("--narrow-workers", type=int, default=NARROW_WORKERS)
    parser.add_argument("--out", type=Path, default=None, help="output directory instead of mnist/sweep, for tests")
    parser.add_argument("--fused", action="store_true",
                        help="run and pilot train each width as one fused model with ntk_lib.train_fused")
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
