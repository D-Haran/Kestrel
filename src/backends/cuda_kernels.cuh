#pragma once
#include <cmath>
#include <cuda_runtime.h>
namespace kestrel::cuda_kernels {
using uint = unsigned int;
struct Params {
    uint n, rows, cols, batch, heads, kv_heads, dim, tokens, position, operation;
    float epsilon, theta;
};
__global__ void matmul(const float *w, const float *x, float *y, Params p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.rows * p.batch)
        return;
    uint r = i % p.rows, b = i / p.rows;
    float sum = 0;
    for (uint c = 0; c < p.cols; ++c)
        sum += w[r * p.cols + c] * x[b * p.cols + c];
    y[i] = sum;
}
__global__ void elementwise(const float *a, const float *b, float *y, Params p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n)
        return;
    if (p.operation == 0)
        y[i] = a[i] + b[i];
    else if (p.operation == 1)
        y[i] = a[i] * b[i];
    else
        y[i] = a[i] / (1 + expf(-a[i]));
}
__global__ void rmsnorm(const float *x, const float *w, float *y, Params p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n)
        return;
    float sum = 0;
    for (uint j = 0; j < p.n; ++j)
        sum += x[j] * x[j];
    y[i] = x[i] * rsqrtf(sum / float(p.n) + p.epsilon) * w[i];
}
__global__ void rope(const float *x, float *y, Params p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.heads * p.dim / 2)
        return;
    uint h = i / (p.dim / 2), d = i % (p.dim / 2), offset = h * p.dim + d;
    float angle = float(p.position) * powf(p.theta, -2.f * float(d) / float(p.dim)),
          c = cosf(angle), s = sinf(angle), a = x[offset], b = x[offset + p.dim / 2];
    y[offset] = a * c - b * s;
    y[offset + p.dim / 2] = b * c + a * s;
}
__global__ void softmax(const float *x, float *y, Params p) {
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= p.rows)
        return;
    uint offset = row * p.n;
    float maximum = -INFINITY;
    for (uint i = 0; i < p.n; ++i)
        maximum = fmaxf(maximum, x[offset + i]);
    float sum = 0;
    for (uint i = 0; i < p.n; ++i) {
        y[offset + i] = expf(x[offset + i] - maximum);
        sum += y[offset + i];
    }
    for (uint i = 0; i < p.n; ++i)
        y[offset + i] /= sum;
}
__global__ void attention_scores(const float *q, const float *k, float *scores, Params p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.heads * p.tokens)
        return;
    uint h = i / p.tokens, t = i % p.tokens, kh = h / (p.heads / p.kv_heads);
    float sum = 0;
    for (uint d = 0; d < p.dim; ++d)
        sum += q[h * p.dim + d] * k[(t * p.kv_heads + kh) * p.dim + d];
    scores[i] = sum / sqrtf(float(p.dim));
}
__global__ void attention_values(const float *probs, const float *v, float *y, Params p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.heads * p.dim)
        return;
    uint h = i / p.dim, d = i % p.dim, kh = h / (p.heads / p.kv_heads);
    float sum = 0;
    for (uint t = 0; t < p.tokens; ++t)
        sum += probs[h * p.tokens + t] * v[(t * p.kv_heads + kh) * p.dim + d];
    y[i] = sum;
}

} // namespace kestrel::cuda_kernels
