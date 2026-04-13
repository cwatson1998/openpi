# OpenPI and LeRobot Interoperability Design

## Executive Summary

This document proposes a concrete architecture for making the local `openpi` codebase interoperate cleanly with LeRobot-based policies, especially SmolVLA, while preserving the existing Pi0.5-oriented workflows already working in `openpi`.

The core recommendation is:

- Keep `openpi` as the main orchestration layer for training entrypoints, evaluation scripts, and policy serving.
- Do not try to unify OpenPI and LeRobot at the checkpoint or model-internals level.
- Introduce a backend abstraction in `openpi` so the same high-level CLI and evaluation code can target either:
  - native OpenPI policies such as `pi0` / `pi0_fast`
  - LeRobot-native policies such as `smolvla` and `pi05`
- Treat environment adaptation and model adaptation as separate concerns.

The practical consequence is that users should be able to say:

```bash
uv run scripts/train.py --backend=openpi --policy=pi0_fast_libero ...
uv run scripts/train.py --backend=lerobot --policy.type=smolvla ...

uv run scripts/serve_policy.py --backend=openpi ...
uv run scripts/serve_policy.py --backend=lerobot --policy.type=smolvla ...

uv run examples/libero/main.py --backend=openpi ...
uv run examples/libero/main.py --backend=lerobot ...
```

without forcing both policy families to share the same checkpoint format, tensor preprocessing implementation, or Python runtime.

## Executive Recommendations

### Recommendation 1

Make `openpi` the owner of:

- CLI surface
- experiment naming
- environment adapters
- evaluation orchestration
- backend selection

Do not make `openpi` the owner of all policy internals.

### Recommendation 2

Define one canonical observation and action schema inside `openpi`, then implement backend adapters:

- `OpenPIBackend`
- `LeRobotBackend`

This is the correct interoperability seam.

### Recommendation 3

Do not target checkpoint interoperability.

OpenPI and LeRobot encode different assumptions into checkpoints:

- OpenPI checkpoints are interpreted through code-defined `TrainConfig` and transform stacks.
- LeRobot checkpoints are interpreted through `from_pretrained(...)` plus saved preprocessor and postprocessor state.

Trying to make those formats interchangeable will create fragile behavior and hard-to-debug mismatches.

### Recommendation 4

Keep preprocessing backend-specific.

Shared orchestration is desirable.
Shared image resize rules, tokenizer behavior, newline prompt handling, normalization ownership, and adapter PEFT target rules are not.

### Recommendation 5

Use a process boundary first for LeRobot integration if dependency skew becomes painful.

This repo already imports an older LeRobot dataset path:

- `lerobot.common.datasets.lerobot_dataset`

Current LeRobot uses newer modules under:

- `lerobot.datasets.*`

That is a signal that a single shared Python environment may become unstable. Start with architecture that permits either:

- in-process backend adapters
- out-of-process backend adapters

without changing the user-facing API.

## Scope

This design focuses on:

- fine-tuning
- evaluation
- policy serving
- runtime interoperability
- API and module boundaries

This design does not attempt:

- direct checkpoint conversion between OpenPI and LeRobot
- reimplementing SmolVLA in JAX
- reimplementing OpenPI Pi0.5 internals in PyTorch

## Current State

## OpenPI Today

OpenPI already has three strong properties that make it a good orchestration layer:

1. It already consumes LeRobot datasets for training.
2. It already has environment-specific evaluation workflows.
3. It already has a compact serving interface centered on `BasePolicy.infer(obs) -> dict`.

### OpenPI Training Shape

Training behavior in OpenPI is controlled by:

- `TrainConfig`
- `DataConfigFactory`
- transform groups
- model config
- checkpoint weight loader

Relevant files:

- [config.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/training/config.py)
- [data_loader.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/training/data_loader.py)
- [train.py](/home/christopher/Documents/openpi-finetune/openpi/scripts/train.py)

Important characteristics:

- datasets are LeRobot-format
- transforms are code-defined, not checkpoint-defined
- normalization stats are loaded from assets and replicated into checkpoints
- model-specific transforms are part of config construction

### OpenPI Serving Shape

Serving behavior is simple:

- load a trained policy from config + checkpoint path
- expose `infer(obs)` over websocket

Relevant files:

- [policy_config.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/policies/policy_config.py)
- [serve_policy.py](/home/christopher/Documents/openpi-finetune/openpi/scripts/serve_policy.py)
- [websocket_policy_server.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/serving/websocket_policy_server.py)

Important characteristics:

- no separate persisted preprocessor object
- transforms are reconstructed from config
- server contract is synchronous and minimal

### OpenPI LIBERO Evaluation Shape

The current LIBERO evaluation loop:

- builds environment observations
- resizes and rotates images in evaluator code
- sends those observations to a policy server
- receives an action chunk
- executes only `replan_steps` before requesting a new chunk

Relevant file:

- [main.py](/home/christopher/Documents/openpi-finetune/openpi/examples/libero/main.py)

This is good for interoperability because the evaluator is already chunk-aware and policy-family-agnostic at the action-chunk level.

## LeRobot Today

LeRobot exposes a more explicit policy runtime stack:

- `PreTrainedPolicy`
- saved model config
- saved model weights
- saved preprocessor pipeline
- saved postprocessor pipeline

Important current references:

- SmolVLA config and policy
- LeRobot training and eval scripts
- LeRobot async policy server

Primary source references:

- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/smolvla/configuration_smolvla.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/smolvla/modeling_smolvla.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/smolvla/processor_smolvla.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/scripts/lerobot_train.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/scripts/lerobot_eval.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/async_inference/policy_server.py

### SmolVLA Fine-Tuning Shape

SmolVLA fine-tuning differs from OpenPI in a few critical ways:

- it is PyTorch-native
- it relies on `PreTrainedPolicy`
- it has explicit processor pipelines
- it uses mean/std normalization for state and action
- it persists processors together with the model

Important SmolVLA config defaults:

- `chunk_size=50`
- `n_action_steps=50`
- `max_state_dim=32`
- `max_action_dim=32`
- image resize with padding to `(512, 512)`
- tokenizer max length `48`
- optimizer defaults different from OpenPI Pi0.5

### SmolVLA Serving Shape

LeRobot async serving does more work inside the serving stack than OpenPI:

1. convert raw robot observations to LeRobot feature schema
2. apply preprocessor
3. predict chunk
4. apply postprocessor per action
5. return timed actions

This is richer than the current OpenPI websocket layer and should not be flattened away.

## Pi0.5 versus SmolVLA

## Why They Feel Similar

At a high level, both are:

- vision-language-action policies
- chunked-action policies
- conditioned on image, state, and task
- suitable for LIBERO-style evaluation

That makes them interoperable at the orchestration layer.

## Why They Should Not Share Internals

They differ in important implementation details:

### OpenPI Pi0.5-like Path

Within this repo, the operative path is closest to:

- `pi0`
- `pi0_fast`

and especially the `pi0_fast_*` fine-tuning and eval paths.

Key properties:

- config-driven transforms
- OpenPI-owned normalization loading
- websocket inference over a plain dict API
- JAX / Flax model execution

### LeRobot SmolVLA Path

Key properties:

- `from_pretrained(...)` object lifecycle
- processor-pipeline lifecycle
- PyTorch-native inference
- action chunking and optional RTC inside policy abstraction

## Interoperability Goal

The correct goal is:

> one user-facing training, evaluation, and serving API in `openpi`, backed by multiple policy runtimes

The wrong goal is:

> one internal implementation of preprocessing, checkpoint loading, and policy execution for both ecosystems

## Design Principles

### Principle 1

Separate environment adaptation from model adaptation.

### Principle 2

Make policy backend selection explicit and first-class.

### Principle 3

Preserve native loading semantics of each backend.

### Principle 4

Preserve native preprocessing semantics of each backend.

### Principle 5

Keep evaluation code backend-neutral after observation construction.

### Principle 6

Make it possible to run backends in separate environments or processes.

## Proposed Architecture

## Layer 1: Canonical Runtime Schema

Introduce a small, explicit canonical schema in `openpi`.

### Canonical Observation

```python
@dataclass
class CanonicalObservation:
    images: dict[str, np.ndarray]
    state: np.ndarray | None
    task: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Canonical Action Chunk

```python
@dataclass
class CanonicalActionChunk:
    actions: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
```

Rules:

- `images` are HWC `uint8` unless a backend explicitly overrides
- `state` is unnormalized robot state in environment coordinates
- `task` is plain text with no backend-specific formatting assumptions
- `actions` are unnormalized environment-space actions

This layer exists only to let env adapters and policy backends communicate cleanly.

## Layer 2: Environment Adapters

Environment adapters own:

- converting env observations into canonical observations
- converting canonical actions into env actions
- episode reset semantics

Examples:

- `LiberoEnvAdapter`
- `DroidEnvAdapter`
- `AlohaEnvAdapter`

These adapters should replace the current partial entanglement between environment schema and model schema in policy-specific transform modules.

### Desired Interface

```python
class EnvAdapter(Protocol):
    def obs_to_canonical(self, raw_obs: Any, task: str | None) -> CanonicalObservation: ...
    def canonical_action_to_env(self, action: np.ndarray) -> Any: ...
    def reset_policy_state(self) -> None: ...
```

## Layer 3: Policy Backends

Policy backends own:

- backend-native checkpoint loading
- preprocessing
- model execution
- postprocessing
- chunking behavior

### Desired Interface

```python
class PolicyBackend(Protocol):
    def reset(self) -> None: ...
    def infer_chunk(self, obs: CanonicalObservation) -> CanonicalActionChunk: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
```

## Backend A: OpenPIBackend

Wrap the existing OpenPI policy object.

Responsibilities:

- build current `openpi.policies.Policy`
- convert canonical observation into the dict shape expected by current OpenPI transforms
- call `infer`
- return `CanonicalActionChunk`

This backend is mostly a compatibility wrapper around existing behavior.

## Backend B: LeRobotBackend

Wrap LeRobot policy loading and processors.

Responsibilities:

- load policy via `PreTrainedPolicy.from_pretrained(...)`
- create preprocessor and postprocessor
- map canonical observation into LeRobot feature dict
- apply backend-native preprocess and postprocess
- return canonical action chunk

This backend preserves native SmolVLA semantics.

## Layer 4: Orchestration API

User-facing CLIs should target:

- environment
- backend
- policy identifier
- checkpoint or pretrained path

not individual framework internals.

### Example CLI Shape

```bash
uv run scripts/serve_policy.py \
  --backend=lerobot \
  --env=LIBERO \
  --policy.type=smolvla \
  --policy.path=/path/to/checkpoint
```

```bash
uv run scripts/serve_policy.py \
  --backend=openpi \
  --env=LIBERO \
  policy:checkpoint \
  --policy.config=pi0_fast_libero_object_train7 \
  --policy.dir=checkpoints/pi0_fast_libero_object_train7/object_train7_full/30000
```

## Recommended Module Layout

Create a new internal package surface in `openpi`:

```text
src/openpi/interoperability/
  canonical_types.py
  env_adapters/
    base.py
    libero.py
    droid.py
    aloha.py
  backends/
    base.py
    openpi_backend.py
    lerobot_backend.py
  serving/
    backend_server.py
  training/
    backend_train.py
```

Optional future expansion:

```text
src/openpi/interoperability/process_boundary/
  protocol.py
  lerobot_worker.py
  client.py
```

## Training Design

## Goal

Allow one `openpi` training entrypoint to dispatch to either:

- native OpenPI training
- LeRobot training

without pretending they are the same trainer.

## Proposed Training Dispatcher

Introduce a thin dispatcher:

```python
class TrainBackend(Protocol):
    def run(self, spec: TrainSpec) -> None: ...
```

### OpenPITrainBackend

Uses existing:

- `TrainConfig`
- `create_data_loader`
- `scripts/train.py`

### LeRobotTrainBackend

Builds or shells out to LeRobot training with:

- policy config
- dataset repo id
- rename map
- output dir
- processor settings

This can start as an adapter that invokes LeRobot programmatically.
If environment conflicts appear, it can become a subprocess boundary later.

## Key Training Recommendation

Do not force one shared config object for both frameworks.

Instead, introduce:

- a shared high-level experiment spec
- backend-specific lowering

### Shared Experiment Spec

```python
@dataclass
class ExperimentSpec:
    backend: Literal["openpi", "lerobot"]
    env: Literal["libero", "droid", "aloha"]
    dataset_repo_id: str
    task_split_file: str | None = None
    task_split: str | None = None
    output_dir: str | None = None
    prompt_source: Literal["dataset_task", "override_file", "default"]
```

Each backend then translates that into its own native config shape.

## Serving Design

## Goal

Support both backends behind one `openpi` serving API.

## Recommendation

Keep the current OpenPI websocket server protocol as the common external API.

Then:

- OpenPI backend serves in-process through current policy objects.
- LeRobot backend is wrapped so it also answers `infer(obs) -> {"actions": ...}`.

This gives you immediate compatibility with:

- `openpi_client`
- current robot examples
- current LIBERO evaluator

## Why Not Reuse LeRobot Async Serving as the Public API

LeRobot async serving is richer, but adopting it as the top-level public API would force:

- new client logic
- new observation envelope semantics
- timed-action streaming semantics
- duplicated evaluation integration

That is unnecessary for initial interoperability.

Instead:

- keep OpenPI websocket as the stable outer contract
- optionally embed or proxy LeRobot async inference internally later

## Evaluation Design

## Goal

Run the same evaluation harness with different policy families.

## Recommendation

Refactor evaluation as:

1. environment adapter creates canonical observation
2. backend produces canonical action chunk
3. evaluator executes first `replan_steps` actions
4. evaluator handles success metrics and logging

This aligns with the current `examples/libero/main.py` flow and requires only one major change:

- stop making the evaluator know policy-specific observation key shapes

## Proposed Evaluation Flow

```text
raw env obs
  -> EnvAdapter.obs_to_canonical(...)
  -> PolicyBackend.infer_chunk(...)
  -> evaluator replans / slices chunk
  -> EnvAdapter.canonical_action_to_env(...)
  -> env.step(...)
```

## Checkpoint and Artifact Strategy

## Non-Goal: Unified Checkpoint Format

Do not attempt to produce one shared checkpoint artifact format.

## Goal: Unified Experiment Registry

Do create a common experiment descriptor format in `openpi` so the user can select:

- backend
- policy type
- checkpoint path
- dataset id
- env
- normalization ownership

Example:

```yaml
name: smolvla_libero_object_train7
backend: lerobot
env: libero
policy_type: smolvla
pretrained_path: checkpoints/smolvla_libero_object_train7/pretrained_model
dataset_repo_id: local/libero_object_train7
task_split_file: data/libero/object_7train_3test.json
task_split: train
```

and similarly:

```yaml
name: pi0_fast_libero_object_train7
backend: openpi
env: libero
policy_config: pi0_fast_libero_object_train7
checkpoint_dir: checkpoints/pi0_fast_libero_object_train7/object_train7_full/30000
dataset_repo_id: local/libero_object_train7
task_split_file: data/libero/object_7train_3test.json
task_split: train
```

## Normalization Strategy

Normalization is one of the main places where subtle regressions will happen.

### Recommendation

Backend owns normalization.

That means:

- OpenPI continues to load and apply its own norm stats from assets/checkpoints.
- LeRobot continues to load and apply its own dataset stats through processor pipelines.

### Anti-Pattern

Do not normalize in the environment adapter.

The environment adapter should stay in raw environment units.

## Prompt Strategy

Prompt handling should also remain backend-owned after canonicalization.

Canonical layer should expose:

- `task: str | None`

Then:

- OpenPI backend can inject default prompt or tokenize per current transform rules.
- SmolVLA backend can apply newline handling and tokenizer rules in its preprocessor.

This avoids silently imposing one prompt formatting policy on both systems.

## Image Strategy

This is another place where backends should differ.

### Shared Rule

Environment adapters should emit raw or lightly standardized `uint8` images in a stable camera dictionary.

### Backend Rule

Each backend should own:

- resize resolution
- padding behavior
- channel ordering fixes
- normalization range
- missing-camera masking rules

This is particularly important because:

- OpenPI Pi0-like paths currently resize to `224x224`
- SmolVLA currently resizes with padding to `512x512`

## Process Model

## Phase 1 Recommendation

Implement interfaces so both in-process and subprocess backends are possible.

Start with:

- `OpenPIBackend`: in-process
- `LeRobotBackend`: in-process if practical

Fallback to:

- `LeRobotSubprocessBackend`

if dependency conflicts become expensive.

## Why This Matters

The local repo currently appears to mix an older LeRobot dataset API with newer LeRobot policy APIs. That is manageable only if architecture does not assume one monolithic dependency graph forever.

## Risks

## Risk 1: Version Skew

OpenPI currently imports:

- `lerobot.common.datasets.lerobot_dataset`

Current LeRobot policy stack uses:

- `lerobot.datasets.*`

This may create import or behavior drift.

### Mitigation

- keep backend boundary explicit
- do not scatter LeRobot imports across unrelated `openpi` modules
- centralize LeRobot integration in one backend package

## Risk 2: Silent Normalization Mismatch

The same dataset may produce different behavior if:

- OpenPI assets are used for SmolVLA
- LeRobot dataset stats are used for OpenPI

### Mitigation

- backend owns normalization
- metadata returned from backend should report normalization source

## Risk 3: Image Semantics Drift

The same observation may be processed differently because of:

- resize target
- pad color
- channel order
- prompt formatting

### Mitigation

- add backend conformance tests on a frozen sample observation
- log preprocessed tensor summaries during debugging

## Risk 4: Chunking Semantics Drift

OpenPI evaluator currently decides how many actions to execute before replanning.
LeRobot policies may also have internal queue assumptions.

### Mitigation

- standardize backend contract on `infer_chunk`
- keep replanning policy in evaluator
- do not call `select_action` from shared evaluator unless the backend explicitly uses that mode

## Risk 5: Over-Unification

The architecture will fail if it tries to erase all backend differences.

### Mitigation

- keep shared layer small
- keep backend-specific lowering explicit

## Testing Strategy

## Unit Tests

Add tests for:

- env adapter output shape
- backend canonical observation mapping
- backend action chunk shape
- reset behavior

## Golden Observation Tests

For one frozen LIBERO observation:

- feed it through OpenPI backend
- feed it through SmolVLA backend
- confirm both produce valid chunk shapes and expected metadata

Do not compare actions numerically across model families.
Compare contract behavior only.

## End-to-End Tests

Add smoke tests for:

- serve OpenPI backend and run simple client
- serve LeRobot backend and run simple client
- run one LIBERO episode with each backend

## Recommended Migration Plan

## Phase 0: No-Behavior-Change Refactor

- Introduce canonical runtime types.
- Introduce env adapter interfaces.
- Wrap current OpenPI policy path in `OpenPIBackend`.
- Keep current CLI behavior.

Success criterion:

- existing Pi0.5-like workflow still works unchanged

## Phase 1: Backend-Neutral Evaluation

- Refactor LIBERO evaluator to use env adapter + backend interface.
- Keep current websocket protocol.
- Support backend selection in evaluation.

Success criterion:

- same evaluator can target OpenPI and LeRobot backend

## Phase 2: LeRobot Serving Integration

- Implement `LeRobotBackend`.
- Add `--backend=lerobot` to `scripts/serve_policy.py`.
- Support SmolVLA checkpoints through `from_pretrained(...)`.

Success criterion:

- current OpenPI client can query SmolVLA through the same outer API

## Phase 3: Training Dispatcher

- Add backend-aware train dispatcher.
- Keep native training implementations underneath.
- Support shared experiment spec lowering to backend-specific configs.

Success criterion:

- one user-facing train entrypoint can dispatch to both ecosystems

## Phase 4: Optional Process Boundary

- add subprocess-backed LeRobot worker if dependency friction persists

Success criterion:

- stable operation without forcing one shared environment

## Concrete Near-Term Implementation Plan

The next implementation cycle should focus on the smallest useful slice.

### Step 1

Create:

- `src/openpi/interoperability/canonical_types.py`
- `src/openpi/interoperability/backends/base.py`
- `src/openpi/interoperability/env_adapters/base.py`

### Step 2

Implement:

- `LiberoEnvAdapter`

by extracting logic from:

- [main.py](/home/christopher/Documents/openpi-finetune/openpi/examples/libero/main.py)

### Step 3

Implement:

- `OpenPIBackend`

as a thin wrapper around current `create_trained_policy(...)`.

### Step 4

Refactor LIBERO evaluation to use:

- `EnvAdapter`
- `PolicyBackend`

with no LeRobot dependency yet.

### Step 5

Implement:

- `LeRobotBackend`

for SmolVLA inference only.

Do not start with LeRobot training integration. Get serving and eval parity first.

## Decisions

## Decision 1

`openpi` remains the top-level orchestrator.

## Decision 2

Environment adaptation and policy adaptation are separate modules.

## Decision 3

Backends own preprocessing, normalization, and checkpoint loading.

## Decision 4

The common external serving contract remains the current OpenPI websocket request-response style.

## Decision 5

Training is unified at the dispatcher level, not at the trainer implementation level.

## Decision 6

Checkpoint interoperability is explicitly out of scope.

## Open Questions

### Question 1

Should LeRobot integration start in-process or as a subprocess immediately?

Proposed answer:

- start in-process if dependency resolution is cheap
- keep subprocess fallback in architecture from day one

### Question 2

Should shared experiment specs live in code or YAML?

Proposed answer:

- start with Python dataclasses
- add YAML only if the number of experiments grows enough to justify registry files

### Question 3

Should OpenPI expose LeRobot `select_action` mode as well as `infer_chunk`?

Proposed answer:

- not initially
- use chunk inference as the stable common contract

## Final Recommendation

Build a small interoperability layer inside `openpi` with:

- canonical observation/action types
- environment adapters
- policy backends
- backend-dispatched serving and evaluation

Keep the shared layer narrow and let each backend remain native below that line.

That design gives you:

- one UX
- one evaluator
- one serving API
- two policy ecosystems

without forcing fragile compatibility at the wrong abstraction level.

## Sources

### Local OpenPI Files

- [config.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/training/config.py)
- [data_loader.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/training/data_loader.py)
- [policy_config.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/policies/policy_config.py)
- [serve_policy.py](/home/christopher/Documents/openpi-finetune/openpi/scripts/serve_policy.py)
- [websocket_policy_server.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/serving/websocket_policy_server.py)
- [libero_policy.py](/home/christopher/Documents/openpi-finetune/openpi/src/openpi/policies/libero_policy.py)
- [main.py](/home/christopher/Documents/openpi-finetune/openpi/examples/libero/main.py)

### LeRobot Primary Sources

- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/factory.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/pretrained.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/smolvla/configuration_smolvla.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/smolvla/modeling_smolvla.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/policies/smolvla/processor_smolvla.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/scripts/lerobot_train.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/scripts/lerobot_eval.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/async_inference/policy_server.py
- https://github.com/huggingface/lerobot/blob/720cf8e3a09f62fa95260cc49a7a30e5d0f7473a/src/lerobot/utils/train_utils.py

### Additional Official References

- https://huggingface.co/lerobot/smolvla_base
- https://huggingface.co/papers/2506.01844
