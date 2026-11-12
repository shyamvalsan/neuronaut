# Receipts for the 4-bit browser model

The explorer can run Qwen3.5-0.8B in the browser: the onnx-community 4-bit export with a `steer` input added after every
MLP (`neuronaut/export_onnx.py`). With a zero push the edited graph gives exactly the original logits. The levers were
validated at full precision, so they were checked again through the exact browser path: the edited ONNX graph, the
float16 unit columns the page uses, greedy decoding.

Design: the 8 levers that held up at full precision; 8 held-out prompts (indices 0, 3, ..., 21 of
`neuronaut/data/heldout_prompts.json`), no system prompt; each lever beside one random direction of the same size and the
unsteered model (136 replies); shuffled blind packets labelled by two independent judges. Data and script in
`docs/q4check/` (`python docs/q4check/tally.py`).

| Lever | Both judges saw the trait | Random push | Unsteered | p (one-sided Fisher) | Readability lever / random / unsteered |
|---|---|---|---|---|---|
| Deity | 7/8 | 0/8 | 0/8 | 0.0007 | 1.4 / 3.4 / 4.0 |
| Reassure | 7/8 | 0/8 | 0/8 | 0.0007 | 1.8 / 2.9 / 4.0 |
| Jesus | 6/8 | 0/8 | 0/8 | 0.004 | 1.3 / 1.6 / 4.0 |
| Anxious | 6/8 | 0/8 | 0/8 | 0.004 | 1.0 / 1.5 / 4.0 |
| Enthusiast | 5/8 | 0/8 | 1/8 | 0.013 | 2.0 / 1.6 / 4.0 |
| Cosmic | 4/8 | 0/8 | 0/8 | 0.04 | 3.0 / 2.8 / 4.0 |
| Gratitude | 3/8 | 0/8 | 0/8 | 0.10 | 2.7 / 3.2 / 4.0 |
| Angry | 2/8 | 0/8 | 0/8 | 0.23 | 2.3 / 3.4 / 4.0 |

Deity, reassure, Jesus and anxious pass p < 0.05/8 on the 4-bit model. Enthusiast and cosmic are nominally significant,
and gratitude and angry cannot be distinguished from chance with 8 prompts. The page shows these numbers next to the
full-precision receipts. Readability is lower than at full precision because replies here are greedy and cut at
48 tokens.
