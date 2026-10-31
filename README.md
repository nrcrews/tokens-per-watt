# tokens-per-watt

How much LLM inference speed does a GPU lose when you cut its power limit? This measures it,
**split into the two halves of inference**:

- **Decode** (writing tokens) streams every weight from memory for each step and does little math
  per byte. The expectation is that it barely notices a lower power limit.
- **Prefill** (reading the prompt) is dense math. The expectation is that it slows roughly with
  the clock that the power limit allows.

Most power-capping results report one blended number. Split, they say something useful: where a
power budget is the constraint, capping every card can serve more tokens from the same megawatt.

> **No measured results yet.** The built-in simulator (`--simulate`) exists to test the pipeline;
> its numbers are made up and every chart it produces is watermarked. Do not quote them.

## What it measures

For each power limit, from the card's maximum down to its minimum:

- decode throughput: short prompts, 512 output tokens each, `ignore_eos`;
- prefill throughput: 4,096-token prompts, one output token each;
- at each concurrency you ask for (default 1 and 32 requests in flight), three repeats;
- energy from the GPU's own cumulative energy counter (NVML), not sampled power;
- SM clock and peak power, sampled every 100 ms.

Prompts are random token IDs, so their length is exact and the prefix cache can't skip prefill.

## Run it

You need a host where you're root and the provider allows power limits (`nvidia-smi -pl`); bare
metal is safest, and some cloud VMs refuse it.

```bash
uv sync
# One engine on the GPU being measured, with prefix caching off.
vllm serve Qwen/Qwen3-8B --no-enable-prefix-caching --max-num-seqs 64 &

sudo .venv/bin/tpw-sweep --gpu 0 --steps 8 --out results.jsonl    # ~35 minutes
.venv/bin/tpw-plot results.jsonl --out charts                     # add --theme dark if you like
```

The sweep puts the card's original power limit back when it finishes, fails, or is interrupted.
SGLang works too: anything serving OpenAI-compatible `/v1/completions` with
`stream_options.include_usage` and `ignore_eos`.

Dry run without a GPU (made-up numbers, watermarked charts):

```bash
uv run tpw-sweep --simulate --duration 2 --repeats 1 --out sim.jsonl && uv run tpw-plot sim.jsonl
```

## The charts

| File | What it shows |
| --- | --- |
| `speed.png` | Decode and prefill throughput as a share of uncapped, against the power limit, with the range where decode keeps 95% shaded |
| `efficiency.png` | Tokens per joule as a share of uncapped |
| `concurrency.png` | Decode at each concurrency: bigger batches do more math, so they feel the cap more |
| `headline.png` | The one number: more decode tokens from the same power budget at the sweet spot |

Every title is computed from the data, so a run that breaks the expected story says so itself.
The **sweet spot** is the lowest limit at which decode, and every higher limit, keeps 95% of its
uncapped speed.

## Demo

On a two-GPU host, `sudo scripts/demo.sh Qwen/Qwen3-8B 350` serves the same model on both cards,
caps GPU 1, and streams one answer from each side by side in tmux with live tokens per second and
watts.

## Caveats

- **GPU power only.** A facility's CPU, memory, networking and cooling are not counted, so
  "tokens per megawatt" is the GPU share of the budget.
- Capping power lowers the clock the card chooses. Locking clocks (`nvidia-smi -lgc`) is a
  different control and isn't swept here.
- Results hold for one card, model, engine version and batch shape. The footer of every chart
  records which.

## Prior art

Power management for LLM inference has been studied before, for example Patel et al.,
*Characterizing Power Management Opportunities for LLMs in the Cloud* (ASPLOS '24), and the
ML.ENERGY / Zeus work on GPU energy. This repo is a small, reproducible measurement that
separates prefill from decode on current hardware.

## License

MIT
