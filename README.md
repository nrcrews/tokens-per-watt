# tokens-per-watt

How much LLM inference speed does a GPU lose when you cut its power limit? This measures it,
**split into the two halves of inference**:

- **Decode** (writing tokens) streams every weight from memory for each step and does little math
  per byte. The expectation is that it barely notices a lower power limit.
- **Prefill** (reading the prompt) is dense math. The expectation is that it slows roughly with
  the clock that the power limit allows.

Most power-capping results report one blended number. Split, they say something useful: where a
power budget is the constraint, capping every card can serve more tokens from the same megawatt.

## Results

Two eight-card H100 machines, one vLLM 0.32.0 engine per card: four cards serving Qwen3-8B,
four serving Qwen3-14B. Each card is normalised to its own uncapped run, and the figures are the
median across a model's four cards, at 32 requests in flight unless noted. Raw measurements are
the `gpu*.jsonl` files in each results folder (`gpu0`–`3` Qwen3-8B, `gpu4`–`7` Qwen3-14B), next to
every chart in light and dark.

Each engine ran as `vllm serve MODEL --no-enable-prefix-caching --max-num-seqs 64
--max-model-len 8192` (image `vllm/vllm-openai:v0.32.0`), and each sweep as `tpw-sweep --steps 8
--concurrency 1,32 --duration 15 --repeats 2 --settle 5`, all eight cards at once.

### H100 SXM (700 W max, 200 W min)

| Power limit | 485 W | 415 W | 345 W | 270 W | 200 W |
| --- | --- | --- | --- | --- | --- |
| Decode speed | 96–99% | 96% | 91–92% | 71% | 51–58% |
| Decode tokens per joule | +5–12% | +23–27% | +41–45% | +41–45% | +29–36% |
| Prefill speed | 83% | 74–75% | 51–56% | 42–44% | 27–28% |
| Prefill tokens per joule | +20% | +25–27% | +3–13% | +9–13% | −5 to −3% |

Ranges span the two models. Down to 415 W (59% of max) decode keeps 96% of its speed for about a
quarter more tokens per joule; its efficiency peaks around 345 W, 8–9% slower. Below that the SM
clock collapses (to about 345 MHz at 200 W) and decode loses speed faster than it saves power.
Uncapped, decode draws only 520–535 W, so limits above that leave it untouched, while prefill
runs at the limit and gets more efficient as it is capped down to 415 W.

![Decode and prefill speed against the power limit, Qwen3-8B on H100 SXM](results/h100-sxm/charts/qwen3-8b/light/speed.png)

Charts: [`results/h100-sxm/charts`](results/h100-sxm/charts).

### H100 PCIe (350 W max, 200 W min)

| At 245 W (70% of max) | Qwen3-8B | Qwen3-14B |
| --- | --- | --- |
| Decode speed, 1 request in flight | 98% | 98% |
| Decode speed, 32 in flight | 96% | 96% |
| Prefill speed | 68% | 68% |
| Decode tokens per joule | +21% | +23% |
| Prefill tokens per joule | −3% | −3% |

Decode never falls off a cliff here: at the 200 W floor it keeps 86–87%, prefill 53%. At the
same 200 W the SXM card runs decode at 345 MHz where the PCIe card holds 900 MHz; the bigger board
appears to spend more of a low limit on things other than the SMs. Prefill gains no efficiency
from the cap on PCIe.
One card, GPU 7, drew only about 307 W during prefill even at the 350 W limit, so its uncapped
prefill baseline is low and it widens the 14B prefill band; without it the median moves by one
point. Charts: [`results/h100-pcie/charts`](results/h100-pcie/charts).

The built-in simulator (`--simulate`) exists to test the pipeline; its numbers are made up and
every chart it produces is watermarked.

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
Pass several results files, one per GPU (`tpw-plot gpu0.jsonl gpu1.jsonl ...`), and each chip is
normalised to its own uncapped run: lines show the median across chips, shading their range.
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

## Development

```bash
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```

## License

MIT
