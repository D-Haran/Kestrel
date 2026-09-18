"""Independent NumPy references for the executable backend kernels."""

import argparse
import json
import subprocess
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--executable", default="build/kestrel_ops")
p.add_argument("--backend", default="cpu")
a = p.parse_args()
j = json.loads(subprocess.check_output([a.executable, a.backend], text=True))
x = np.array(j["x"])
w = np.array(j["w"])
expected = dict(
    rmsnorm=x / np.sqrt(np.mean(x * x) + 1e-5) * w,
    silu=x / (1 + np.exp(-x)),
    swiglu=x / (1 + np.exp(-x)) * w,
    softmax=np.exp(x - x.max()) / np.exp(x - x.max()).sum(),
    add=x + w,
    matmul=np.array(j["matrix"]).reshape(3, 16) @ x,
)
z = x.reshape(2, 8)
angle = 7 * 10000.0 ** (-np.arange(4) * 2 / 8)
c = np.cos(angle)
s = np.sin(angle)
expected["rope"] = np.concatenate(
    [z[:, :4] * c - z[:, 4:] * s, z[:, 4:] * c + z[:, :4] * s], axis=-1
).ravel()
q = x.reshape(4, 4)
k = np.array(j["k"]).reshape(3, 2, 4)
v = np.array(j["v"]).reshape(3, 2, 4)
out = []
for h in range(4):
    score = k[:, h // 2] @ q[h] / 2
    prob = np.exp(score - score.max())
    prob /= prob.sum()
    out.extend(prob @ v[:, h // 2])
expected["attention"] = out
for name, value in expected.items():
    np.testing.assert_allclose(j[name], value, atol=2e-6, rtol=2e-5, err_msg=name)
    print(name, "passed")
