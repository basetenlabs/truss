# Changelog

## Unreleased

### Internal refactor: consolidate push argument set into `PushOptions`

Resolves the TODO from Tyron in `truss/remote/baseten/remote.py` about the set of
user-intent flags being drilled through `BasetenRemote.push`, `_prepare_push`,
and `create_truss_service`.

#### What changed

**New:** `PushOptions` Pydantic model in `truss/remote/baseten/custom_types.py`.
Holds the full set of user-intent flags that used to be passed as positional
kwargs through every layer: `publish`, `promote`,
`preserve_previous_prod_deployment`, `disable_truss_download`, `deployment_name`,
`origin`, `environment`, `deploy_timeout_minutes`, `team_id`, `labels`,
`preserve_env_instance_type`, `include_git_info`.

**Moved and slimmed:** `FinalPushData` moved from `remote.py` to
`custom_types.py`. Removed the duplicated fields (`preserve_previous_prod_deployment`,
`origin`, `environment`, `allow_truss_download`, `team_id`, `labels`) and replaced
them with a single `options: PushOptions` reference. `is_draft` and
`allow_truss_download` are now derived `@property` accessors on the options.

**Internal signatures:**
- `BasetenRemote._prepare_push(truss_handle, model_name, options, progress_bar)`
- `create_truss_service(api, push_data, truss_user_env, semver_bump)`

**Public surfaces unchanged:**
- `truss.api.push(**kwargs)` — same signature, same kwarg names, same defaults
- `BasetenRemote.push(**kwargs)` — same signature; constructs `PushOptions`
  internally
- `BasetenRemote.push_chain_atomic(**kwargs)` — same; constructs `PushOptions`
  internally for its call to `_prepare_push`

**No wire-format change:** every inner API method (`api.create_model_from_truss`,
`api.create_model_version_from_truss`, `api.create_development_model_from_truss`)
is called with identical kwargs before and after. The GraphQL payload is
byte-identical. No base image rebuild, no user-facing config migration, no
change to deployed-model behavior.

#### Why these specific choices

**Why keep kwargs on `BasetenRemote.push` and `truss.api.push`?**
Both functions are part of the documented SDK. External callers (Baseten users
writing Python scripts) construct push calls with named kwargs. Changing to
`push(options: PushOptions)` would be a breaking change for every script that
calls `truss.push(target_dir, publish=True, environment="staging")`. The
duplication cost of marshalling kwargs into `PushOptions` inside these two
functions is contained and acceptable; the backwards-compatibility benefit is
not negotiable for a public SDK.

**Why make `create_truss_service` take `FinalPushData` (option "4b")?**
Originally we considered leaving its signature alone and having `push()` unpack
the struct when calling it. We chose the full refactor because
`create_truss_service` has exactly one non-test caller (`BasetenRemote.push`),
and every test call-site was going to be rewritten anyway as part of the
`FinalPushData` reshape. Taking the slightly bigger change in one atomic PR is
cleaner than leaving the internal surface half-migrated with a follow-up ticket
that might never get picked up.

**Why is `PushOptions` frozen, and why does `normalize()` return a new
instance?**
Previously `_prepare_push` mutated its `publish` and `environment` parameters in
place based on model-server type and promote/environment interactions. That
mutation was easy to miss when reading the code. A frozen struct plus an
explicit `options = options.normalize(model_server)` line at the top of
`_prepare_push` makes the mutation impossible to overlook, and makes it unit
testable in isolation (see `test_push_options.py`).

**Why did some validation move to `PushOptions` but `disable-truss-download`
validation did not?**
The pure validators — `deploy_timeout_minutes` range, `deployment_name`
character set, `preserve_previous_prod_deployment` requires `promote` — only
need the options themselves and nothing else, so they moved to field validators
and a model validator on `PushOptions`. This lets them fail at construction time
rather than after an S3 upload has started.

The `disable-truss-download can only be used for new models` check requires an
API lookup (`exists_model`) to determine whether the model name is new. It
stayed in `_prepare_push` because `PushOptions` must not reach out to a network
service during construction.

**Why default `PushOptions.publish` to `False`?**
Pre-existing inconsistency in the codebase: `truss.api.push` (public SDK)
defaults `publish=False`, but `BasetenRemote.push` defaults `publish=True`.
`PushOptions` must pick one default for direct construction
(`PushOptions()`-style callers). We chose `False` to match the public SDK
contract, since that is the documented default. Internal code that wants
publishing must opt in explicitly — which is the safer default: a typo or
missing field produces a draft deployment (recoverable) rather than an
unintended published one (harder to reverse).

**Why is `include_git_info` on `PushOptions` even though it doesn't get drilled
through downstream?**
`include_git_info` is only read inside `BasetenRemote.push` itself (to pick
between `TrussUserEnv.collect()` and `TrussUserEnv.collect_with_git_info()`).
Strictly speaking it does not solve the duplicated-args problem Tyron flagged.
We included it in `PushOptions` anyway for consistency: all user-intent flags
live in one place, so future readers do not have to remember which knobs are in
the struct and which are loose kwargs.

**Why did `push_chain_atomic` keep its local `publish = True` mutation before
constructing `PushOptions`?**
`push_chain_atomic` iterates over multiple chainlet artifacts and calls
`_prepare_push` for each. Each call normalizes independently based on the
artifact's model-server. By computing the `publish` upgrade upfront once
(matching current behavior exactly) we avoid any chance of different chainlets
making different normalization decisions. Preserves existing semantics with
zero behavior drift.

#### Risks evaluated and accepted

- **`FinalPushData` field-shape change.** Verified via grep that no code outside
  `truss/remote/baseten/` constructs or destructures `FinalPushData`, so
  removing fields is safe.
- **Validator error messages.** Kept verbatim from the previous `_prepare_push`
  versions so existing `pytest.raises(ValueError, match="...")` assertions
  continue to pass with no regex changes.
- **Subclasses of `BasetenRemote`.** Verified via grep that no subclasses exist
  in this repo or its siblings (`truss-chains`, `truss-train`), so internal
  signature changes are safe.
- **Smoketests.** Verified smoketest scripts construct `BasetenRemote` but do
  not call `.push()`, so no smoketest contract depends on internals.

#### Test coverage

- New `truss/tests/remote/baseten/test_push_options.py` — 20 unit tests
  covering field validators, model validator, `normalize()` rules, and frozen
  semantics.
- Updated `test_core.py` with a `_make_push_data` helper to keep each test's
  setup minimal.
- Updated mock assertions in `test_remote.py`, `test_cli.py`, and
  `test_chain_upload.py` to read through `kwargs["push_data"].options.X`
  instead of direct kwargs on the `create_truss_service` mock.

Full test suite for `truss/tests/remote/` and `truss/tests/cli/`: 367 passed.
