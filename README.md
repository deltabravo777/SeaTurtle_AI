# SeaTurtle_AI

SeaTurtle AI is an experimental language-model architecture.

The project explores whether a model can learn useful sequence structure with a lightweight field-style architecture instead of relying heavily on full attention. The reference implementation combines causal lane-field propagation, grouped convolutional resonance, per-lane memory decay, and a small attention polish layer.

## Core idea

Most transformer language models rely on attention as the main way tokens communicate.

SeaTurtle AI explores a different path:

```text
tokens
→ embedding
→ causal lane-field memory
→ grouped Conv1d resonance
→ per-lane decay/write dynamics
→ lightweight attention polish
→ language-model logits
