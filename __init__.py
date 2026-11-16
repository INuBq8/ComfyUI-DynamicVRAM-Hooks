"""
ComfyUI-DynamicVRAM-Hooks
=========================

Experimental monkey patch: lets LoRA hooks (CreateHookLora / SetClipHooks /
PairConditioningSetProperties ...) run on ModelPatcherDynamic instead of
falling back to a full non-dynamic copy of the model.

How stock ComfyUI handles hooks under dynamic VRAM (0.45.x):
  * CFGGuider.inner_set_conds swaps the model for get_non_dynamic_delegate(),
    a full-load clone, whenever a cond carries hooks.
  * SetClipHooks clones the CLIP with disable_dynamic=True.
  * The non-dynamic patcher bakes each hook group's LoRA into the weights
    (patch_hooks) and restores them afterwards. With USDU this setup is paid
    on every tile.

What this patch does instead:
  * Hook LoRAs are NOT baked into weights. For every weight a registered hook
    touches, a small function is added to the module's weight_function /
    bias_function list. comfy.ops runs those functions on the cast weight on
    every forward, after the dynamic-VRAM resident weight (with the normal,
    static LoRAs already merged) has been fetched.
  * patch_hooks() only records which hook group is active (on the BaseModel),
    so switching hook groups between cond batches costs nothing.
  * The function copies the weight, adds the active group's LoRA delta, and
    returns the copy, so the cached VRAM weight is never modified.
  * The delegate swap and the CLIP disable_dynamic clone are skipped.

Disable with the environment variable COMFY_DYNVRAM_HOOKS=0.
"""

import logging
import os
import weakref

import torch

log = logging.getLogger("DynamicVRAM-Hooks")
PREFIX = "[DynamicVRAM-Hooks]"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_ATTR_ACTIVE = "_dvh_active"      # ModelPatcher attr: (HookGroup, {key: [patch tuples]}) or None
_ATTR_DEVCACHE = "_dvh_devcache"  # BaseModel attr: {id(cpu_tensor): (cpu_tensor, gpu_tensor)}
_ATTR_INSTALLED = "_dvh_keys"     # BaseModel attr: set of keys with an installed function


def _enabled_by_env():
    return os.environ.get("COMFY_DYNVRAM_HOOKS", "1").strip().lower() not in ("0", "false", "no", "off")


# --------------------------------------------------------------------------
# device mirroring for hook patch tensors (LoRA up/down etc.)
# --------------------------------------------------------------------------

def _to_device(value, device, cache):
    """Mirror comfy.lora.prefetch_prepared_value: map every tensor inside a patch
    value onto `device`, caching by id so each LoRA tensor is uploaded once."""
    import comfy.weight_adapter as weight_adapter

    if isinstance(value, torch.Tensor):
        if value.device == device:
            return value
        hit = cache.get(id(value))
        if hit is not None and hit[0] is value:
            return hit[1]
        moved = value.to(device=device, non_blocking=True)
        cache[id(value)] = (value, moved)  # keep the source alive so its id stays unique
        return moved
    if isinstance(value, weight_adapter.WeightAdapterBase):
        return type(value)(value.loaded_keys, _to_device(value.weights, device, cache))
    if isinstance(value, tuple):
        return tuple(_to_device(v, device, cache) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device, cache) for v in value]
    return value


# --------------------------------------------------------------------------
# per-weight function installed on comfy ops modules
# --------------------------------------------------------------------------

class HookWeightFunction:
    """Applied by comfy.ops after the weight is cast to the compute device."""

    _dvh_marker = True

    def __init__(self, model, key):
        self.model_ref = weakref.ref(model)
        self.key = key

    def __call__(self, weight):
        model = self.model_ref()
        if model is None:
            return weight
        patcher = _active_patcher(model)
        if patcher is None:
            return weight
        active = _active_patches(patcher)
        if not active:
            return weight
        patches = active.get(self.key)
        if not patches:
            return weight

        import comfy.lora

        cache = getattr(model, _ATTR_DEVCACHE, None)
        if cache is None:
            cache = {}
            setattr(model, _ATTR_DEVCACHE, cache)
        patches = [_to_device(p, weight.device, cache) for p in patches]

        import comfy.float
        import comfy.utils

        # Same math as stock ModelPatcher.patch_hook_weight_to_device: fp32 copy,
        # calculate_weight, stochastic rounding back with the key's seed.
        # (calculate_weight works in place; the copy also protects the resident VRAM weight.)
        out = weight.to(dtype=torch.float32, copy=True)
        out = comfy.lora.calculate_weight(patches, out, self.key)
        if weight.dtype != torch.float32:
            out = comfy.float.stochastic_rounding(out, weight.dtype, seed=comfy.utils.string_to_seed(self.key))
        return out


def _active_patcher(model):
    """The patcher whose hook state applies right now.

    Diffusion models: BaseModel.current_patcher (set by pre_run for the patcher
    being sampled), so clones sharing the same weights never see each other's
    hooks. Text encoders have no current_patcher: use the patcher that last
    called patch_hooks on this model."""
    patcher = getattr(model, "current_patcher", None)
    if patcher is not None:
        return patcher
    ref = getattr(model, "_dvh_owner", None)
    return ref() if ref is not None else None


def _active_patches(patcher):
    """{key: patch list} for the patcher's current hook group, or None.

    current_hooks is the single source of truth; the combined dict is cached
    on the patcher and rebuilt lazily (e.g. for a clone that inherited
    current_hooks but not the cache)."""
    hooks = getattr(patcher, "current_hooks", None)
    if hooks is None:
        return None
    cached = getattr(patcher, _ATTR_ACTIVE, None)
    if cached is not None and cached[0] is hooks:
        return cached[1]
    combined = patcher.get_combined_hook_patches(hooks=hooks)
    setattr(patcher, _ATTR_ACTIVE, (hooks, combined))
    return combined


def _resolve_module(model, key):
    import comfy.utils

    parts = key.rsplit(".", 1)
    if len(parts) != 2:
        return None, None
    try:
        module = comfy.utils.get_attr(model, parts[0])
    except AttributeError:
        return None, None
    return module, parts[1]


def _install_functions(patcher):
    """Make sure every key that any registered hook patches has a HookWeightFunction.

    comfy's dynamic load() rebuilds weight_function lists on every load, so this is
    idempotent and safe to call repeatedly."""
    model = patcher.model
    keys = set()
    for hook_ref_patches in patcher.hook_patches.values():
        keys.update(hook_ref_patches.keys())
    if not keys:
        return 0

    installed = 0
    skipped = []
    for key in keys:
        module, param = _resolve_module(model, key)
        if module is None or param not in ("weight", "bias") or not hasattr(module, "comfy_cast_weights"):
            skipped.append(key)
            continue
        attr = param + "_function"
        current = getattr(module, attr, [])
        if any(getattr(f, "_dvh_marker", False) and getattr(f, "key", None) == key for f in current):
            continue
        # CastWeightBiasOp.weight_function is a class-level list: always assign a new one.
        setattr(module, attr, list(current) + [HookWeightFunction(model, key)])
        if getattr(module, "_combined_linear", None) is not None:
            log.warning(f"{PREFIX} {key} belongs to a CombinedLinear group; hook LoRA may be ignored on the fused path.")
        installed += 1

    if skipped:
        seen = getattr(model, "_dvh_warned", set())
        new = [k for k in skipped if k not in seen]
        if new:
            log.warning(f"{PREFIX} {len(new)} hooked keys are not comfy ops weights and will not be hooked, e.g. {new[:3]}")
            seen.update(new)
            model._dvh_warned = seen
    setattr(model, _ATTR_INSTALLED, keys)
    return installed


# --------------------------------------------------------------------------
# patched ModelPatcherDynamic methods
# --------------------------------------------------------------------------

def _apply():
    import comfy.model_patcher as mp
    import comfy.hooks

    MPD = getattr(mp, "ModelPatcherDynamic", None)
    if MPD is None:
        log.warning(f"{PREFIX} comfy.model_patcher.ModelPatcherDynamic not found; nothing patched.")
        return False
    if getattr(MPD, "_dvh_patched", False):
        return True
    for name in ("load", "patch_hooks", "get_non_dynamic_delegate", "add_hook_patches"):
        if not hasattr(MPD, name):
            log.warning(f"{PREFIX} ModelPatcherDynamic.{name} missing; this ComfyUI version is not supported. Nothing patched.")
            return False

    orig_load = MPD.load
    orig_add_hook_patches = MPD.add_hook_patches

    def load(self, *args, **kwargs):
        result = orig_load(self, *args, **kwargs)
        try:
            _install_functions(self)
        except Exception:
            log.exception(f"{PREFIX} installing hook functions after load failed")
        return result

    def add_hook_patches(self, hook, patches, strength_patch=1.0, strength_model=1.0):
        # Hook LoRAs never touch the staged weights here, so keep patches_uuid:
        # a new uuid would mark the model dirty and re-stream every weight.
        old_uuid = self.patches_uuid
        keys = orig_add_hook_patches(self, hook, patches, strength_patch=strength_patch, strength_model=strength_model)
        self.patches_uuid = old_uuid
        try:
            _install_functions(self)
        except Exception:
            log.exception(f"{PREFIX} installing hook functions failed")
        return keys

    def patch_hooks(self, hooks):
        # Recompute even for the same group: keyframe changes alter hook.strength.
        setattr(self, _ATTR_ACTIVE, None)
        self.model._dvh_owner = weakref.ref(self)
        if hooks is not None:
            _install_functions(self)
            combined = self.get_combined_hook_patches(hooks=hooks)
            for key, plist in combined.items():
                for p in plist:
                    v = p[1]
                    if isinstance(v, tuple) and len(v) == 2 and v[0] == "model_as_lora":
                        log.warning(f"{PREFIX} model_as_lora patches are not supported ({key}).")
            setattr(self, _ATTR_ACTIVE, (hooks, combined))
        self.current_hooks = hooks

    # unpatch_hooks stays the stock no-op: nothing is baked into weights, and
    # dynamic load() calls it, which must not drop the active group mid-sampling.
    # patch_hooks(None) / clean_hooks() clear the state instead.

    def clear_cached_hook_weights(self):
        self.cached_hook_patches.clear()
        self.patch_hooks(None)
        cache = getattr(self.model, _ATTR_DEVCACHE, None)
        if cache:
            cache.clear()

    def get_non_dynamic_delegate(self):
        # Stay dynamic: hooks are handled by HookWeightFunction.
        return self

    MPD.load = load
    MPD.add_hook_patches = add_hook_patches
    MPD.patch_hooks = patch_hooks
    MPD.clear_cached_hook_weights = clear_cached_hook_weights
    MPD.get_non_dynamic_delegate = get_non_dynamic_delegate
    MPD._dvh_patched = True

    # SetClipHooks forces a non-dynamic CLIP clone (reloads the text encoder per node).
    try:
        # comfy_extras are imported by file path before custom nodes, so the class the
        # server uses lives in NODE_CLASS_MAPPINGS, not necessarily in comfy_extras.nodes_hooks.
        import nodes as comfy_nodes

        targets = []
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("SetClipHooks")
        if cls is not None:
            targets.append(cls)
        try:
            import comfy_extras.nodes_hooks as nodes_hooks
            if nodes_hooks.SetClipHooks not in targets:
                targets.append(nodes_hooks.SetClipHooks)
        except Exception:
            pass

        def apply_hooks(self, clip, schedule_clip, apply_to_conds, hooks=None):
            if hooks is not None:
                clip = clip.clone()
                if apply_to_conds:
                    clip.apply_hooks_to_conds = hooks
                clip.patcher.forced_hooks = hooks.clone()
                clip.use_clip_schedule = schedule_clip
                if not clip.use_clip_schedule:
                    clip.patcher.forced_hooks.set_keyframes_on_hooks(None)
                clip.patcher.register_all_hook_patches(hooks, comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Clip))
            return (clip,)

        for t in targets:
            t.apply_hooks = apply_hooks
        if not targets:
            log.warning(f"{PREFIX} SetClipHooks not found; CLIP hooks keep the stock non-dynamic path.")
    except Exception:
        log.exception(f"{PREFIX} could not patch SetClipHooks; CLIP hooks keep the stock non-dynamic path.")

    return True


try:
    import comfy.cli_args

    if not _enabled_by_env():
        log.info(f"{PREFIX} disabled by COMFY_DYNVRAM_HOOKS=0")
    elif not comfy.cli_args.enables_dynamic_vram():
        log.info(f"{PREFIX} dynamic VRAM is off; nothing to patch.")
    elif _apply():
        log.info(f"{PREFIX} active: LoRA hooks run on dynamic VRAM (per-forward weight functions, no non-dynamic delegate).")
except Exception:
    log.exception(f"{PREFIX} failed to apply; stock behaviour is unchanged.")
