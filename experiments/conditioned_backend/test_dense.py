import sys, time, numpy as np
sys.path.insert(0, "/tmp/cb")
from harness import capture, run_lane, dense_check, step_plan
for name in sys.argv[1:]:
    circuit, sampler, got = capture(name)
    ops = got[-1]
    plan = step_plan(ops)
    rng = np.random.default_rng(0)
    worst = 0; peak = 0; nrand = 0
    t = time.time()
    for shot in range(40):
        f = rng.integers(0, 2, ops.num_f) if shot % 2 else np.zeros(ops.num_f, int)
        f = f * (rng.random(ops.num_f) < 0.05)
        tr = run_lane(ops, plan, f, rng)
        peak = max(peak, max(w for *_, w in tr)); nrand += sum(1e-12 < p < 1 - 1e-12 for _, p, _, _ in tr)
        if ops.num_lines <= 16:
            worst = max(worst, dense_check(ops, f, tr))
    print(f"{name}: lines {ops.num_lines} ops {ops.num_ops} steps {plan.num_steps} layers {len(plan.layer_ops)} applied {len(plan.apply_ops)} | peak width {peak} random outcomes {nrand} | max |p - p_dense| {worst:.2e} | {time.time()-t:.1f}s", flush=True)
