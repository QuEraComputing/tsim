import sys, time, numpy as np
sys.path.insert(0, "/tmp/cb")
from harness import capture, packed, concrete_signs, step_plan
from tsim.statevector.conditioned import lane as L
import tsim.statevector.conditioned.through as T
from tsim.statevector.conditioned.gauss import layer_amplitude
circuit, sampler, got = capture("blp")
ops = got[-1]; plan = step_plan(ops)
n, W, ox, oz, oe = packed(ops)
s = concrete_signs(ops, np.zeros(ops.num_f, np.int64)); theta = np.where(s & 1, -ops.angle, ops.angle)
rng = np.random.default_rng(1)
frame, active, pos, amps, cache, cver = L.new_state(n, W, ops.num_ops)
target_step = int(sys.argv[1])
for step in range(target_step):
    m = int(plan.measure[step]); layer = plan.layer(step)
    p1, keep, phi, kv = T.prepare_through(frame, active, pos, amps, cache, cver, ox, oz, oe, layer, theta[layer].copy(), m, W, 26)
    if len(phi):
        amps = T.finish_through(active, pos, keep, phi, kv, int(rng.random() < p1))
    amps = L.compress(frame, active, pos, amps, W)
m = int(plan.measure[target_step]); layer = plan.layer(target_step); ang = theta[layer].copy()
w = L.width_of(amps)
def bench(f, reps=300):
    f(); t = time.perf_counter()
    for _ in range(reps): f()
    return 1e6 * (time.perf_counter() - t) / reps
eps, eta, s4 = T._factor_signs(ang)
rows, m_row = T._layer_rows(frame, cache, cver, ox, oz, oe, layer, m, W)
print(f"step {target_step} w {w} L {len(layer)}")
print(f" layer_rows     {bench(lambda: T._layer_rows(frame, cache, cver, ox, oz, oe, layer, m, W)):7.2f} us")
print(f" images         {bench(lambda: T._images(rows, s4, m_row, active, w, n, W)):7.2f} us")
zimg, ximg = T._images(rows, s4, m_row, active, w, n, W)
print(f" frozen_basis   {bench(lambda: T._frozen_basis(zimg, pos, active, w, W)):7.2f} us")
basis, count = T._frozen_basis(zimg, pos, active, w, W)
def fr():
    f2, z2, x2, r2, m2, b2 = frame.copy(), zimg.copy(), ximg.copy(), rows.copy(), m_row.copy(), basis.copy()
    return T._freeze(f2, z2, x2, r2, m2, b2, count, active, pos, w, W)
print(f" freeze (+copies) {bench(fr):7.2f} us  count {count}")
f2, z2, x2, r2, m2, b2 = frame.copy(), zimg.copy(), ximg.copy(), rows.copy(), m_row.copy(), basis.copy()
keep = T._freeze(f2, z2, x2, r2, m2, b2, count, active, pos, w, W)
print(f" image_of_zero  {bench(lambda: T._image_of_zero(z2, keep, r2, m2, eps, eta, W)):7.2f} us  wa {len(keep)}")
psi0, support = T._image_of_zero(z2, keep, r2, m2, eps, eta, W)
print(f" image_of_reg   {bench(lambda: T._image_of_register(amps, x2, keep, psi0, support, W)):7.2f} us  support 2^{int(np.log2(len(support)))}")
print(f" layer_amplitude{bench(lambda: layer_amplitude(r2, len(layer), eps, eta, m2, np.zeros(W, np.uint64), W)):7.2f} us")
def full():
    f2, a2, p2, c2, cv2 = frame.copy(), active.copy(), pos.copy(), cache.copy(), cver.copy()
    return T.prepare_through(f2, a2, p2, amps, c2, cv2, ox, oz, oe, layer, ang, m, W, 26)
print(f" prepare_through (+copies) {bench(full):7.2f} us")
