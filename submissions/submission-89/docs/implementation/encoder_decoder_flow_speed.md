# Encoder–Decoder & Flow‑Speed System

> Detailed reference for the **speed‑conditioned encoder–decoder** stack (`SpeedEncDec`, `SpeedLayerNormPow`) and the **flow‑speed prediction** modules used by the GMM Transformer framework.

---

## 1  Motivation

Traditional transformer pipelines first embed **discrete tokens** into a latent space. In point‑cloud/GMM settings, inputs are **continuous coordinates**. We therefore:

* Replace fixed embeddings with an **orthogonal encoder–decoder pair** that can learn rotations and (optionally) learnable scaling while preserving invertibility.
* Introduce a **flow‑speed \( s\in[0,1] \)** that adaptively modulates the *magnitude* of residual updates. Higher‑quality (high SNR) inputs require minor updates, lower‑quality samples require stronger corrections.

---

## 2  Speed‑Conditioned Encoder–Decoder

### 2.1  Architecture

```
        X ∈ ℝ^{B×N×d}                 (raw coordinates)
               │
               ▼
    ┌────────────────────┐   U,V = orthogonal matrices
    │   SpeedEncDec.encode │   B₀  = null‑space basis
    └────────────────────┘   σ    = learnable log‑scale
               │ z ∈ ℝ^{B×N×D}
               ▼
        Transformer …        (contextual processing)
               │ h
               ▼
    ┌────────────────────┐
    │   SpeedEncDec.decode │  (uses same parameters)
    └────────────────────┘
               │ Ŷ  (normalised prediction)
```

`SpeedEncDec(d, D)` holds parameters:

| Symbol | Shape | Role |
|--------|-------|------|
| \(U\) | \(D×d\) | projects input into latent space |
| \(V\) | \(d×d\) | rotation in original space |
| \(B_0\) | \(d×D\) | projects latent null‑space back to input dim |
| \(\log σ ∈ ℝ^{d}\) | (optional) per‑dimension scale parameters |
| \(A ∈ 𝔰𝔬(D)\) | skew‑symmetric matrix for additional rotation (currently inactive) |

All linear factors are **orthogonal** by parametrisation with Torch's Cayley map → preserves norm stability.

### 2.2  Mathematics

Encoding (current implementation):
\[
 z = X U^⊤ \tag{1}
\]

Decoding:
1. Project back to data space: \( P = z U \)
2. Rotate via \( V \): \( Y = P V^⊤ \)
3. Recover null‑space component
\[
 z_{⊥} = z - P U^⊤,\qquad N = z_{⊥} B_0^⊤
\]
4. Output:
\[
 \widehat X = Y + N \tag{2}
\]

**Identity when \(s=0\)** – scaling & rotation branches are multiplied by speed‑dependent factors that vanish at zero (see commented code for potential extensions).

### 2.3  Properties

* Invertible under full parameterisation when scale term is active.
* Lipschitz‑stable due to orthogonality.
* Guarantees \(\widehat X=X\) at speed 0 → curriculum learning friendly.

---

## 3  Speed‑Conditioned Layer Normalisation

`SpeedLayerNormPow` provides a *soft transition* between identity and standard LayerNorm:

\[
\text{SLN}(x,t) = γ(t)\;x + β(t),\qquad t\in[0,1]
\]
where
\[
 γ(t) = \exp(t\;γ),\qquad β(t) = t\;β.
\]
Here \(γ,β\) are learnable vectors. At \(t=0\) the operation is identity; at \(t=1\) it equals an affine re‑scaling. (The usual centring/variance terms are currently skipped to keep variance information intact.)

---

## 4  Flow‑Speed Prediction

Flow‑speed \(s\) modulates residual connections inside each `TransformerBlock`:
\[
 Y = X + s\;Δ(X),\qquad 0≤s≤1. \tag{3}
\]
### 4.1  Interfaces

```python
flow = predictor(targets, inputs)  # returns Tensor[B] or Tensor[B,L]
```
* **Global flow** – shape \([B]\)
* **Per‑layer flow** – shape \([B,L]\)

### 4.2  Predictor Implementations

| Class | Idea | Equation |
|-------|------|----------|
| `DummyFlowPredictor` | Always 1 | \(s=1\) |
| `LinearFlowPredictor` | Affine map on SNR (clipped) | \(
 s = s_{min} + (s_{max}-s_{min})\;\frac{\text{SNR}_{max}-\text{SNR}}{\text{SNR}_{max}-\text{SNR}_{min}}\)
| `MonotonicFlowPredictor` | Learn monotone spline (piece‑wise linear) | see Eq. (4) |

Piece‑wise spline (per layer or shared):
\[
 s(x)=h_i+\frac{h_{i+1}-h_i}{t_{i+1}-t_i}(x-t_i),\qquad x∈[t_i,t_{i+1}) \tag{4}
\]
with learned non‑negative height deltas \(Δh_i\) guaranteeing monotonicity and normalised so \(s\in[0,1]\).

### 4.3  Fractional Flow Distribution

When layers are **virtually repeated** (cycle/layerwise/grouped), flow can be *fractionally* distributed:

\[
 s_j = \begin{cases}
 1,& j < ⌊k s⌋\\
 k s - ⌊k s⌋,& j = ⌊k s⌋\\
 0,& \text{otherwise}
 \end{cases} \tag{5}
\]
where \(k\) = repetition factor, \(j\) = current replica index. This enables sub‑unit application counts (e.g., effective 0.3 passes).

---

## 5  Interaction Summary

1. **Encoder/Norm/Decoder** read **local** flow (scalar per sample) – typically the average of per‑layer flow.
2. **Transformer blocks** consume (possibly per‑layer) flow via Eq. (3).
3. Predictors map **observable difficulty** (SNR) → flow in a *monotone* fashion, ensuring that better data gets gentler updates.

---

## 6  Key Advantages

* **Curriculum‑friendly** – identity path at \(s=0\) stabilises early training.
* **Efficiency** – dynamic residual strength reduces wasted computation on easy examples.
* **Extensibility** – drop‑in new predictor with minimal code changes. 