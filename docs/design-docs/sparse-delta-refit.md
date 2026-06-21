# Sparse Delta vLLM Refit

Sparse refit is for non-colocated Megatron policy and vLLM generation workers.
The same delta-compression config supports two transports:

- NCCL collective refit (`backend: vllm` with the collective synchronizer) sends
  the first update as a dense full sync, then frames sparse delta payloads over
  the existing model-update process group.
- HTTP refit (`backend: vllm_http` or `refit_transport="vllm_http_sparse"`) is
  for clusters or regions where NCCL collectives are not viable.

Sender and receiver must load the same initial checkpoint. The sender keeps a
CPU baseline, sends sparse additive deltas, then commits the new baseline after
the receiver update path succeeds.

For cross-region HTTP, use `backend: vllm_http` for the GRPO path, or call
`create_weight_synchronizer(..., refit_transport="vllm_http_sparse", ...)` when
an integration owns receiver URLs. This is not a dense checkpoint transport; a
dense refresh requires reloading the receiver from the matching checkpoint.

## Config

Sender:

```yaml
backend: vllm_http
colocated: {enabled: false}
urls: [http://vllm-refit-relay.nemo-rl.svc.cluster.local:8080]
api_key_env_var: NRL_REFIT_API_KEY
request_timeout_s: 600
delta_compression:
  enabled: true
  dtype: bf16
  full_sync_interval: 1000000
  sparse_bucket_size_bytes: 268435456
  delta_load_batch_size_bytes: 1073741824
  index_encoding: indices
```

Receiver:

```yaml
backend: vllm
vllm_cfg:
  expose_http_refit_server: true
  http_refit_server_port: 8081
  http_refit_api_key_env_var: NRL_REFIT_API_KEY
```

`expose_http_refit_server` exposes receiver endpoints only; it does not change
an in-process `backend: vllm` GRPO run away from NCCL refit.

## Wire Contract

NCCL collective sparse refit frames each broadcast as a small header followed by
a packed payload. Header `kind=full` carries dense tensors for the first sync or
configured full refreshes. Header `kind=delta` carries sparse metadata plus the
packed indices/value tensors. vLLM post-load hooks run after full updates; sparse
delta updates apply directly to mapped tensors and use additive loader fallback
for unsupported tensors.

HTTP sparse refit normalizes base URLs and calls receiver refit endpoints:

- `GET /nemo-rl/refit/health`
- `POST /nemo-rl/refit/sparse-delta`
- `POST /nemo-rl/refit/flush`

The separate `vllm_http` generation backend posts token-id batches to
`/nemo-rl/generate` when its configured generation service exposes that route.

Sparse buckets use `sparse_indices` on both transports. Contiguous updates use
`range_start` plus value count and omit packed indices for that tensor.
Non-contiguous updates default to int32 absolute positions. For lower-bandwidth
links, set `delta_compression.index_encoding` to `deltas` for uint16/uint32
gap-encoded positions, or `deltas_zstd` to additionally compress those position
bytes when the `zstandard` package is installed. For HTTP refit only, set
`NRL_REFIT_HTTP_BODY_COMPRESS=zlib|zstd` on the sender to compress the serialized
sparse request body before it crosses the region boundary. Sparse values are
always additive deltas; this keeps NCCL and HTTP receiver semantics identical.
If configured,
`x-nemo-rl-refit-key` carries the internal API key; still restrict access with
cluster networking, service mesh policy, or VPC routing.

## Lifecycle

Initial setup builds the source baseline while the receiver already holds the
checkpoint. The NCCL path sends the dense first sync and then queues baseline
prewarm after the source broadcast so the copy can overlap inference-side
collective drain and generation work. For `vllm_http`, GRPO can overlap remote
baseline initialization with step-0 generation. Later syncs invalidate reusable
vLLM caches, wait for baseline prewarm, encode sparse buckets, update receivers,
and commit the source baseline.

Useful knobs: `NRL_REFIT_HTTP_POST_PARALLELISM`,
`NRL_REFIT_HTTP_INFLIGHT_BUCKETS`, `NRL_REFIT_HTTP_POOL_MAXSIZE`,
`NRL_REFIT_HTTP_BODY_COMPRESS`, `NRL_REFIT_ASYNC_RECEIVER_APPLY`,
`NRL_REFIT_RECEIVER_APPLY_QUEUE_DEPTH`,
`NRL_REFIT_PREWARM_DELTA_BASELINE`, `NRL_REFIT_BASELINE_IN_MEMORY`,
`NRL_REFIT_BASELINE_MMAP_DIR`, and `NRL_REFIT_DIRECT_SPARSE_VLLM_LOAD`.

## Validation

Verify both sides loaded the same checkpoint, health succeeds, baseline prewarm
or initialization is logged, post-update refit logs `REFIT_HTTP_TIMING`, async
apply logs `REFIT_RECEIVER_TIMING`, generation waits for sparse refit completion
or observes a newer weight version, no dense full-sync error appears, and the
chosen `index_encoding` is tested on the same topology before adopting it.
