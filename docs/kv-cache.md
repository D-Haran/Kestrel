# KV cache

`KVCache` separates insertion, prefix access, length, capacity, memory accounting, and reset. `ContiguousKVCache` allocates keys and values separately in `[layer, capacity, kv_head, head_dim]` order. Each layer has its own committed length. Append requires the exact next position and full K/V vectors; out-of-order inserts and over-capacity writes throw.

Allocated memory is `2 * layers * capacity * kv_heads * head_dim * sizeof(float)`, or O(L·T·Hkv·D). The factor of two is K and V. GQA reduces cache storage relative to full multi-head attention by `kv_heads / heads`. Allocation arithmetic is checked for overflow.

For the pretrained model with capacity 47, the cache is `2*6*47*6*48*4 = 649728` bytes. CLI defaults to the checkpoint context limit; benchmarks allocate exactly prompt length plus N−1 decode positions. Reset clears lengths, not backing bytes; old entries cannot be accessed through valid prefix spans. This is logical reset, not secure erasure.

A future paged cache can replace insertion and accounting behind the interface, but the current attention interface still consumes contiguous spans. A paged implementation would need to materialize a prefix or extend the backend attention contract with block tables. Kestrel does not claim to implement paged attention.
