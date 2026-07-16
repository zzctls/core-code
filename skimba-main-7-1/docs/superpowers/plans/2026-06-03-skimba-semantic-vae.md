# Skimba Semantic VAE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `skimba-main` VAE architecture match the 20-class semantic VAE trained by `Code_VAE_SSC`, so pretrained VAE weights can be loaded and frozen in Skimba.

**Architecture:** Keep the existing Skimba latent diffusion training path intact. Change only the VAE boundary so `AutoEncoderKL`, its encoder, decoder, and builder accept `num_class` plus optional `semantic_embed_dim`, matching `Code_VAE_SSC/stable_diffusion/models/autoencoder_MAE_Class_13_2.py`.

**Tech Stack:** Python, PyTorch, strict YAML config loading with `strictyaml`, source-level `pytest` tests for architecture compatibility.

---

## File Structure

- Modify `skimba-main/stable_diffusion/models/autoencoder.py`: add semantic VAE constructor parameters, pass them through encoder and decoder, and make decoder logits class-count configurable.
- Modify `skimba-main/builder/model_builder_3D_Voxel_unet_diffusion.py`: read `num_class` and optional `semantic_embed_dim` from config and pass them into `AutoEncoderKL`.
- Modify `skimba-main/config/config.py`: allow optional `semantic_embed_dim` in strict model config schema.
- Modify `skimba-main/config/semantickitti_autoencoder.yaml`: set `num_class: 20` and add `semantic_embed_dim: 8`.
- Modify `skimba-main/stable_diffusion/modules/distributions.py`: clamp `log_var` to match `Code_VAE_SSC`.
- Create `skimba-main/tests/test_semantic_vae_architecture.py`: source-level tests that do not import CUDA, `spconv`, or dataset modules.

---

### Task 1: Add Failing Architecture Compatibility Tests

**Files:**
- Create: `skimba-main/tests/test_semantic_vae_architecture.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_semantic_vae_architecture.py` with:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_autoencoder_exposes_codevae_semantic_constructor_boundary():
    source = read_source("stable_diffusion/models/autoencoder.py")

    assert "num_class=2" in source
    assert "semantic_embed_dim=None" in source
    assert "self.num_class = num_class" in source
    assert "self.semantic_embed_dim = semantic_embed_dim or num_class" in source
    assert "num_class=self.num_class" in source
    assert "semantic_embed_dim=self.semantic_embed_dim" in source


def test_autoencoder_encoder_and_decoder_match_codevae_semantic_shapes():
    source = read_source("stable_diffusion/models/autoencoder.py")

    assert "input_channels = semantic_embed_dim + 3" in source
    assert "nn.Linear(input_channels, 16)" in source
    assert "FinalConv(num_outs=num_class" in source
    assert "FinalConv(num_outs=2" not in source
    assert "nn.Linear(4, 16)" not in source


def test_diffusion_builder_passes_semantic_vae_config_to_autoencoder():
    source = read_source("builder/model_builder_3D_Voxel_unet_diffusion.py")

    assert "num_class = model_config['num_class']" in source
    assert "semantic_embed_dim = model_config.get('semantic_embed_dim', None)" in source
    assert "num_class=num_class" in source
    assert "semantic_embed_dim=semantic_embed_dim" in source


def test_config_schema_and_default_config_accept_semantic_embedding():
    schema_source = read_source("config/config.py")
    config_source = read_source("config/semantickitti_autoencoder.yaml")

    assert "Optional(\"semantic_embed_dim\"): Int()" in schema_source
    assert "num_class: 20" in config_source
    assert "semantic_embed_dim: 8" in config_source


def test_gaussian_distribution_clamps_log_variance_like_codevae():
    source = read_source("stable_diffusion/modules/distributions.py")

    assert "torch.clamp(self.log_var" in source
    assert "-20" in source
    assert "10" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_semantic_vae_architecture.py -q
```

Expected: tests fail because `skimba-main` still has `FinalConv(num_outs=2)`, `nn.Linear(4, 16)`, no optional `semantic_embed_dim` schema entry, and no active `log_var` clamp.

---

### Task 2: Update VAE Architecture Boundary

**Files:**
- Modify: `skimba-main/stable_diffusion/models/autoencoder.py`
- Test: `skimba-main/tests/test_semantic_vae_architecture.py`

- [ ] **Step 1: Add semantic parameters to encoder and decoder builders**

In `autoencoder.py`, change `build_encoder` to:

```python
def build_encoder(
    in_channels,
    latent_channels,
    autoencoder_channels_list,
    autoencoder_num_res_blocks,
    groups,
    dropout_rate,
    num_class=2,
    semantic_embed_dim=None,
):
    return Encoder(
        in_channels=in_channels,
        out_channels=latent_channels,
        channels_list=autoencoder_channels_list,
        num_res_blocks=autoencoder_num_res_blocks,
        groups=groups,
        dropout_rate=dropout_rate,
        num_class=num_class,
        semantic_embed_dim=semantic_embed_dim,
    )
```

Change `build_decoder` to:

```python
def build_decoder(
    latent_channels,
    out_channels,
    autoencoder_channels_list,
    autoencoder_num_res_blocks,
    groups,
    in_channels,
    dropout_rate,
    num_class=2,
):
    return Decoder(
        in_channels=latent_channels,
        out_channels=out_channels or in_channels,
        channels_list=autoencoder_channels_list,
        num_res_blocks=autoencoder_num_res_blocks,
        groups=groups,
        dropout_rate=dropout_rate,
        num_class=num_class,
    )
```

- [ ] **Step 2: Add semantic fields to `AutoEncoderKL`**

Add constructor parameters:

```python
num_class=2,
semantic_embed_dim=None,
```

Then store and pass them:

```python
self.num_class = num_class
self.semantic_embed_dim = semantic_embed_dim or num_class

self.encoder = build_encoder(
    self.in_channels,
    self.latent_channels,
    self.autoencoder_channels_list,
    self.autoencoder_num_res_blocks,
    self.groups,
    dropout_rate,
    num_class=self.num_class,
    semantic_embed_dim=self.semantic_embed_dim,
)
self.decoder = build_decoder(
    self.latent_channels,
    self.out_channels,
    self.autoencoder_channels_list,
    self.autoencoder_num_res_blocks,
    self.groups,
    self.in_channels,
    dropout_rate,
    num_class=self.num_class,
)
```

- [ ] **Step 3: Make `Encoder` consume semantic embedding width**

Add constructor parameters:

```python
num_class=2,
semantic_embed_dim=None,
```

Then set:

```python
self.num_class = num_class
semantic_embed_dim = semantic_embed_dim or num_class
self.semantic_embed_dim = semantic_embed_dim
input_channels = semantic_embed_dim + 3
```

Keep:

```python
self.conv_input = nn.Sequential(
    nn.Linear(input_channels, 16),
    nn.ReLU()
)
```

- [ ] **Step 4: Make `Decoder` output semantic logits**

Add constructor parameter:

```python
num_class=2
```

Then set:

```python
nclasses = num_class
self.out_conv = FinalConv(num_outs=num_class, out_channels=channels * 2)
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_semantic_vae_architecture.py -q
```

Expected: architecture tests that inspect `autoencoder.py` pass except builder/config/distribution tests.

---

### Task 3: Pass Semantic VAE Config Through Diffusion Builder

**Files:**
- Modify: `skimba-main/builder/model_builder_3D_Voxel_unet_diffusion.py`
- Test: `skimba-main/tests/test_semantic_vae_architecture.py`

- [ ] **Step 1: Read semantic config values**

Near existing model parameter reads, add:

```python
num_class = model_config['num_class']
semantic_embed_dim = model_config.get('semantic_embed_dim', None)
```

- [ ] **Step 2: Pass values into `AutoEncoderKL`**

Update the constructor call to include:

```python
num_class=num_class,
semantic_embed_dim=semantic_embed_dim,
```

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m pytest tests/test_semantic_vae_architecture.py -q
```

Expected: builder test passes; config and distribution tests may still fail.

---

### Task 4: Update Strict Config Schema And Default Config

**Files:**
- Modify: `skimba-main/config/config.py`
- Modify: `skimba-main/config/semantickitti_autoencoder.yaml`
- Test: `skimba-main/tests/test_semantic_vae_architecture.py`

- [ ] **Step 1: Allow optional semantic embedding config**

Change the `strictyaml` import to include `Optional`:

```python
from strictyaml import Bool, Float, Int, Map, Optional, Seq, Str, as_document, load
```

Add this field after `"num_class": Int(),` in `model_params`:

```python
Optional("semantic_embed_dim"): Int(),
```

- [ ] **Step 2: Update default Skimba VAE config**

In `config/semantickitti_autoencoder.yaml`, replace:

```yaml
num_class: 2 #20
```

with:

```yaml
num_class: 20
semantic_embed_dim: 8
```

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m pytest tests/test_semantic_vae_architecture.py -q
```

Expected: config tests pass; distribution test may still fail.

---

### Task 5: Match Code_VAE_SSC VAE Distribution Stability

**Files:**
- Modify: `skimba-main/stable_diffusion/modules/distributions.py`
- Test: `skimba-main/tests/test_semantic_vae_architecture.py`

- [ ] **Step 1: Clamp VAE log variance**

Replace the commented clamp with:

```python
self.log_var = torch.clamp(self.log_var, -20, 10)
```

- [ ] **Step 2: Run tests**

Run:

```powershell
python -m pytest tests/test_semantic_vae_architecture.py -q
```

Expected: all tests in `tests/test_semantic_vae_architecture.py` pass.

---

### Task 6: Verify State Dict Shape Compatibility Against Code_VAE_SSC Source

**Files:**
- No source changes expected.

- [ ] **Step 1: Compare architecture source boundaries**

Run:

```powershell
Select-String -Path "stable_diffusion/models/autoencoder.py" -Pattern "num_class=2","semantic_embed_dim=None","input_channels = semantic_embed_dim + 3","FinalConv(num_outs=num_class"
Select-String -Path "..\Code_VAE_SSC\stable_diffusion\models\autoencoder_MAE_Class_13_2.py" -Pattern "num_class=2","semantic_embed_dim=None","input_channels = semantic_embed_dim + 3","FinalConv(num_outs=num_class"
```

Expected: both files expose the same semantic constructor defaults and class-logit boundary.

- [ ] **Step 2: Run final focused verification**

Run:

```powershell
python -m pytest tests/test_semantic_vae_architecture.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Report remaining runtime caveat**

Report that architecture-level compatibility is verified locally. Full checkpoint loading still depends on the actual `Code_VAE_SSC` semantic VAE checkpoint being trained with the same config values, especially `semantic_embed_dim: 8`, `num_class: 20`, and the same `autoencoder.py` layer names.
