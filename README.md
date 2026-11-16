# ComfyUI-DynamicVRAM-Hooks

An experimental monkey patch that makes ComfyUI's **LoRA hooks** (masked/scheduled LoRAs: `CreateHookLora`, `SetClipHooks`, `PairConditioningSetProperties`, hook keyframes...) run on **Dynamic VRAM**. Without it, ComfyUI falls back to a slow non-dynamic copy of the model.

In a real workflow (Anima 2.9B, 2 masked hook LoRAs, Ultimate SD Upscale with 22 tiles, RTX 4090), the run time went from **466 s to 203 s**, with the same images.

> **This is a workaround, not the proper fix.** It patches ComfyUI internals from outside. The real fix belongs in ComfyUI's backend, written by someone who maintains the model patcher and comfy-aimdo code. This repo documents the problem and shows one approach that works.

Related upstream issues: [Comfy-Org/ComfyUI#12853](https://github.com/Comfy-Org/ComfyUI/issues/12853) (hooks broke when Dynamic VRAM was introduced) and [Comfy-Org/ComfyUI#14382](https://github.com/Comfy-Org/ComfyUI/issues/14382) (hook crash on quantized models and the hook slowdown).

---

## The problem

### Measurements

Same workflow, same machine (RTX 4090 24 GB, 128 GB RAM, ComfyUI 0.45.0, comfy-aimdo 0.6.0, `--high-ram`):

| | Dynamic VRAM ON | Dynamic VRAM OFF |
|---|---|---|
| No hooks | 196 s | 201 s |
| 2 hook LoRAs (stock ComfyUI) | **466 s** | 280 s |
| 2 hook LoRAs (**with this patch**) | **203 s** | n/a |

- **Dynamic VRAM itself costs nothing.** Without hooks, ON and OFF are the same.
- **Hooks alone cost about 40%** (201 → 280 s).
- **Hooks with Dynamic VRAM cost 138%** (196 → 466 s).

The cost is paid **once per sampler run**, not per step. Ultimate SD Upscale starts the sampler once per tile, so the cost is multiplied:

| Seconds per 896×1152 tile (4 steps) | Time |
|---|---|
| No hooks | ~3.9 s |
| Hooks, Dynamic VRAM OFF | ~5.5 s |
| Hooks, Dynamic VRAM ON (stock) | ~11.4 s |
| Hooks, Dynamic VRAM ON (patched) | ~4.2 s |

A plain workflow with one sampler run hides this almost completely. That is probably why it has gone largely unnoticed.

### Root cause

Stock hooks work by **rewriting the model weights**: before each hook group's batch, `ModelPatcher.patch_hooks()` merges that group's LoRA into the weights, and `unpatch_hooks()` restores them afterwards.

Dynamic VRAM has no stable full-model weights to rewrite. Weights are streamed on demand and cached in comfy-aimdo's VRAM pool. So in ComfyUI 0.45.x:

- `ModelPatcherDynamic.patch_hook_weight_to_device()` just raises *"Hooks not implemented in ModelPatcherDynamic"*.
- To avoid that, `CFGGuider.inner_set_conds()` (`comfy/samplers.py`) swaps the model for `get_non_dynamic_delegate()` whenever a cond has hooks. That is a **full-load, non-dynamic clone** of the model.
- `SetClipHooks` (`comfy_extras/nodes_hooks.py`) clones the CLIP with `disable_dynamic=True`, which **reloads the text encoder once per SetClipHooks node**.
- Every sampler run then pays for setting up the delegate and patching hook weights. With USDU that happens on every tile.

---

## The fix (what this node does)

The key observation is in `comfy/ops.py`: every comfy ops layer has `weight_function` / `bias_function` lists. On **every forward**, after Dynamic VRAM has fetched the weight from its VRAM cache (with normal, non-hook LoRAs already merged), the cast weight is passed through those functions. They run on a value derived from the cached weight, so the cache itself is never modified. That is exactly the hook point needed.

The patch changes the following:

1. **Per-layer hook functions.** For every weight that a registered hook touches, a small `HookWeightFunction` is added to that layer's `weight_function` / `bias_function`. When the layer runs, it looks up the **currently active hook group**:
   - If there is none, it returns the weight unchanged.
   - If there is one, it applies that group's LoRA to an **fp32 copy** of the weight with `comfy.lora.calculate_weight`, then rounds back with `comfy.float.stochastic_rounding` using the key's seed. This is the same math as stock `patch_hook_weight_to_device`, which is why the images are identical.
2. **`ModelPatcherDynamic.patch_hooks()`** only records which hook group is active. ComfyUI already calls it before each hook group's batch, so switching groups costs nothing. Keyframe changes still work, because `prepare_hook_patches_current_keyframe` calls `patch_hooks(None)` and the combined patches are rebuilt on the next apply.
3. **The active group is tracked per patcher.** The function reads it from `BaseModel.current_patcher` (set by `pre_run`). Text encoders have no `current_patcher`, so for them it uses the patcher that last called `patch_hooks` on that model. Clones that share the same weights therefore never see each other's hooks.
4. **`get_non_dynamic_delegate()` returns `self`**, so no non-dynamic copy is made.
5. **`SetClipHooks` clones the CLIP without `disable_dynamic=True`**, so the text encoder stays dynamic and is not reloaded per node.
6. **`add_hook_patches()` keeps `patches_uuid` unchanged.** Hook LoRAs no longer touch the staged weights, so marking the model dirty, which would re-stream every weight, is unnecessary.
7. **The functions are reinstalled after every dynamic `load()`**, because load rebuilds the `weight_function` lists.
8. **LoRA tensors are uploaded to the GPU once** and cached, instead of being copied over on every forward.

This also sidesteps the `'Linear' object has no attribute 'weight_scale'` crash (#14382) for hooks on quantized models, because `get_key_patches()` is never called. The quantized case has not been tested, though; see the limitations below.

### Cost

Applying a LoRA delta per layer per forward is cheap next to the forward pass itself. A rank-32 delta is about 1% of the layer's compute, plus memory traffic for the temporary fp32 copy. In the workflow above, hook overhead dropped from +138% to about +3%.

---

## Verification

All tests ran on a second ComfyUI instance driven through the HTTP API, with the same prompts and seeds, comparing **patched** against **stock** (`COMFY_DYNVRAM_HOOKS=0`) pixel by pixel:

| Test (Anima base, 1 MP, 30 steps) | Result |
|---|---|
| 3 masked regions, 2 hook LoRAs, seed 42 | pixel-identical |
| 3 masked regions, 2 hook LoRAs, seed 43 | pixel-identical |
| 2 masked regions, 1 hook LoRA | pixel-identical |
| No-hook run right after a hooked run (checks that hooks don't leak) | pixel-identical |
| Plain single-region run | pixel-identical |
| Six back-to-back 4-step sampler runs (simulating tiles) | identical; **6.75 s → 4.1 s per run** |

In the real USDU workflow, the hook LoRAs were confirmed visually to be applied in the patched run.

---

## Install

```
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-DynamicVRAM-Hooks
```

Restart ComfyUI. With Dynamic VRAM enabled (the default), the startup log shows:

```
[DynamicVRAM-Hooks] active: LoRA hooks run on dynamic VRAM (per-forward weight functions, no non-dynamic delegate).
```

The node adds no nodes to the menu; it only changes behaviour. With `--disable-dynamic-vram` it does nothing.

### Disable without uninstalling

Set the environment variable `COMFY_DYNVRAM_HOOKS=0` before starting ComfyUI. For example, add this line to your `.bat` launcher:

```
set COMFY_DYNVRAM_HOOKS=0
```

---

## Limitations and caveats

- **Version-specific.** Written against ComfyUI **0.45.0** and comfy-aimdo **0.6.0**. It checks that the methods it replaces exist, but not that their behaviour is unchanged. After a ComfyUI update, check the startup line and compare one hooked image with `COMFY_DYNVRAM_HOOKS=0`.
- **Tested on bf16 Anima models only** (Anima base and an Anima 2.9B finetune).
  - **Quantized models (fp8 / int8 / ConvRot):** the weight is dequantized before the hook is applied, which should work but is untested.
  - **CombinedLinear:** int8 layers fused into a CombinedLinear group take a fast path that skips `weight_function`. Hooks on those layers would be ignored; the patch logs a warning if it finds one.
- **Unsupported patch types and layers:**
  - `model_as_lora` patch types are not supported in hooks; a warning is logged.
  - Hooked weights that are not comfy ops layers (raw `nn.Parameter`s outside comfy ops) are skipped with a warning. LoRAs normally target only linear and conv layers.
- **Single GPU only.** The multi-GPU clone path was not tested.
- **Not an official fix.** A proper implementation in ComfyUI could integrate this with comfy-aimdo's caching and memory accounting, which a monkey patch cannot do.

---

## Files

- `__init__.py`: the whole patch, with comments explaining each replaced method.
