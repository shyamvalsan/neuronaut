# Receipts

Find single neurons that change how a small language model behaves, watch what they do in 3D, and test them against
chance before you believe them.

Pushing one MLP neuron can make Qwen3.5-0.8B talk like a deity, panic, or flatter you. It is also easy to fool yourself:
any push changes the text, and with enough tries every neuron looks meaningful. Receipts makes the honest version the
easy one. Every lever it reports comes with receipts: a frozen test on prompts it never saw, beside random pushes of
the same size, judged blind.

```
pip install neuron-receipts
receipts try deity "How do vaccines work?"
```

```
[unsteered]
Vaccines work by stimulating the immune system to produce a specific response to a harmless virus or bacteria ...

[deity  L18·N2521 +0.6]
Vaccines work by using **divine deceptiveness** to trigger a **divine response** in the body ...

[random direction, same layers and size]
Vaccines are powerful, scientifically proven ways to protect the immune system ...

Receipts: trait seen in 23/24 blind-judged replies vs 0/24 for a random push (p = 8e-13)
```

## What it does

| Command | What happens |
|---|---|
| `receipts try LEVER "prompt"` | The unsteered model, the lever, and a random push of the same size, side by side. `LEVER` is a name from the recipe or any neuron, like `L18N2521:+0.6` or `L18N2521:0.4+L16N157:-0.5`. |
| `receipts list` | The built-in levers with their receipts. |
| `receipts serve` | The 3D explorer in your browser: the model as a tower of 24 layers and 86,016 neurons. Click a neuron for its portrait (what wakes it up, what it writes, its wiring), push it, and watch the ripple travel up the tower beside a random push. |
| `receipts find NAME --words a,b,c` | Search for candidate levers: push every neuron in the chosen layers on short contexts, keep those that raise your words more than every random direction does, then sweep the best with generated text beside random controls. Or `--contrast pairs.jsonl` with texts that do and do not show the trait. |
| `receipts test candidates.json` | The frozen test: writes the settings and a hash before any output, runs 24 new prompts under three system prompts, greedy and sampled, beside random directions and the unsteered model, asks 12 easy questions to measure the damage, and has two independent judges label a shuffled sample blind. Writes receipts into the recipe and a report page. |
| `receipts atlas` | Precomputes the explorer's data for a model (once). |
| `receipts site` | Exports a static explorer for GitHub Pages, with recorded outputs, measured ripples, and an in-browser engine. |

Runs on a laptop CPU: float32 PyTorch, no GPU, no gradients. Apple's GPU or CUDA are used if present.

## The 3D explorer

- **The tower.** One disc per layer, bottom to top in the order the model runs them; inside a disc, neurons that
  write in similar directions sit close together. Brightness is how active a neuron usually is.
- **A neuron's portrait.** The words it writes (both signs), the text where it fires hardest, and its strongest direct
  wiring to earlier and later neurons, drawn as threads. Wiring is read from the weights (how much one neuron's
  typical activity shifts another's input through the residual stream). It leaves out attention, so it is labelled
  as wiring, a lead, not a measurement.
- **The ripple.** Push a neuron and the tower shows, layer by layer, how far the push moves the model's state at the
  last prompt token and which neurons change most, then plays a random push of the same size for comparison.
- **Share and record.** Copy a link to the exact neuron, lever, push size and prompt; record a ten-second square clip.

In the browser with no install, the explorer can run the model itself: Qwen3.5-0.8B in 4 bits on WebGPU (about 650 MB on
first visit). It uses the official onnx-community export with one change, a `steer` input added after every MLP
(`receipts/export_onnx.py`); with a zero push it matches the original graph exactly, and in spot checks the levers show the same behaviour as
full precision. The 4-bit model is slightly different from the one the receipts come from, so its own receipts are
measured separately (see `docs/`).

## Built-in levers (Qwen3.5-0.8B)

From the study in [qwen-neuron-levers](https://github.com/shyamvalsan/qwen-neuron-levers): frozen before testing,
24 new prompts × 3 system prompts, a random direction of the same size for each, two blind judges.

| Lever | Neuron, push | Both judges saw the trait | Random push |
|---|---|---|---|
| Deity | L18 N2521 +0.6 | 23/24 | 0/24 |
| Reassurer | L14 N2022 +0.6 | 22/24 | 0/24 |
| Jesus | L16 N157 −0.9 | 20/24 | 0/24 |
| Anxious | L19 N606 +0.9 | 18/24 | 0/24 |
| Enthusiast | L12 N2730 −0.9 | 17/24 | 0/24 |
| Flatterer | L15 N2912 −0.6 | 13/24 | 0/24 |
| Cosmic narrator | L15 N3502 −0.6 | 11/24 | 0/24 |
| Angry | L13 N2994 +0.6 | 10/24 | 0/24 |
| Grandiose, India, Human life story | | 7, 3 and 0 of 24 | did not hold up |

The cost: at these push sizes the model answers easy questions much worse (10/12 unsteered, a median of 5.5/12 under
random pushes of the same size, 0 to 8/12 under the levers). Most of the damage comes from the size of the push, not
from which neuron is pushed.

## What a push is

Qwen3.5's MLPs are SwiGLU: `h = SiLU(W_gate x) * (W_up x)`, `out = W_down h`. Pushing neuron *n* of layer *L* adds

```
dose * scale[L] * W_down[:, n] / ||W_down[:, n]||
```

to that layer's MLP output at every position, which is exactly the neuron firing harder (tested in
`tests/test_engine.py`). `scale[L]` is the median size of the model's state leaving layer *L*, so a dose of 0.6 is 0.6
times the typical size there. Weights are never changed.

## Models

Built and tested on Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B`, revision `2fc0636`). The engine works with other Hugging Face
models whose MLPs have a `down_proj` (Qwen3, Llama, Gemma, Mistral style; checked on a tiny Llama in the tests). The
built-in recipe and atlas are for Qwen3.5-0.8B; other models need `receipts atlas` and their own levers.

## Judging

`receipts test` uses two independent Claude judges through the Anthropic API (credentials from the environment, as the
`anthropic` package resolves them). Each judge sees the replies shuffled in its own order, with no condition labels.
`--judge export` writes the blind packets instead, for any judges you prefer; `receipts tally RUN` reads their labels.
Label files that are byte-identical between the two judges are rejected, because that means one judge's labels were
copied, not produced independently.

## Credits

The method follows [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) by lyra (relative push sizes,
random-direction controls, the steer lens) and Goodfire's
[Neural Networks Are Already Clean](https://already-clean.goodfire.com/). The "evil Qwen" recipes that started this are
from [ziva.sh](https://ziva.sh/blogs/evil-qwen). For a null result on neuron labels in small Qwen models, which is why
every lever here needs receipts, see [Batutin](https://andriybatutin.substack.com/p/we-tried-to-confirm-neural-networks).
The browser runtime is [ONNX Runtime Web](https://onnxruntime.ai/) with the
[onnx-community](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX) export; the tokenizer is
[Transformers.js](https://github.com/huggingface/transformers.js); the 3D view is [three.js](https://threejs.org/).

MIT licence. Shyam Sreevalsan, 2026.
