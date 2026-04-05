# ⚡ Aether — Hashcat Rule Performance Benchmark Tool

> Scientific-grade Hashcat rule benchmarking with GPU acceleration and advanced data visualizations.

---

## Overview

**Aether** (`aether.py`) is a high-performance benchmarking and analysis tool for [Hashcat](https://hashcat.net/hashcat/) rule files. It executes rules against a test word corpus on the GPU via **OpenCL**, measures execution time with statistical precision, ranks rules by performance, and produces publication-quality visualizations — inspired by the Michelson-Morley precision measurement methodology.

It supports the **complete Hashcat rule grammar**: simple 1-character rules, 2-character positional rules, 3-character substitution/manipulation rules, and all reject/filter rules.

---

## Features

- 🔬 **Full Hashcat Rule Engine** — implements all rule transformations (`l`, `u`, `c`, `r`, `d`, `s`, `T`, `D`, `x`, `i`, `o`, `*`, `^`, `$`, `@`, `!`, `/`, reject rules, and more) in an OpenCL kernel
- ⚡ **GPU Acceleration** — processes words × rules in parallel via PyOpenCL
- 🎯 **Identical-Dicts Precision Mode** — tiles the built-in word list to eliminate word-variance noise when benchmarking reject and filter rules
- 📊 **Advanced Visualizations**
  - Radar charts (speed, throughput, consistency)
  - Performance heatmaps per rule-type character
  - Statistical summary dashboards (distribution, scatter, bar, pie)
  - Distribution plots with KDE density curves and box plots
- 📁 **Sorted Output** — writes a new `.rule` file with rules ordered fastest-first
- 📄 **JSON Reports** — per-file performance reports with full statistical metrics (mean, median, std dev, CV%, min/max, ops/sec)
- 🧬 **Rule Set Optimizer** — merges reports from multiple files and selects the fastest subset within configurable time and count constraints
- 🗂️ **Directory Scanning** — recursively discovers `.rule` files from paths

---

## Requirements

### Python ≥ 3.8

| Package | Purpose |
|---|---|
| `pyopencl` | GPU kernel execution |
| `numpy` | Numerical operations & data preparation |
| `matplotlib` | Chart generation |
| `seaborn` | Enhanced plot styling |
| `pandas` | Data manipulation |
| `scipy` *(optional)* | KDE density curves in distribution plots |

Install all dependencies:

```bash
pip install pyopencl numpy matplotlib seaborn pandas scipy
```

> **Note:** `pyopencl` requires an OpenCL-capable GPU and the appropriate platform drivers (NVIDIA CUDA, AMD ROCm, Intel OpenCL, or Apple Metal).

---

## Installation

```bash
git clone https://github.com/yourname/aether.git
cd aether
pip install -r requirements.txt
```

Or just drop `aether.py` into your working directory alongside your `.rule` files.

---

## Usage

```bash
python3 aether.py -r <rule_file_or_dir> [options]
```

### Minimal Example

```bash
python3 aether.py -r best64.rule
```

### With Visualizations

```bash
python3 aether.py -r best64.rule --visualize
```

### Limit Rules for a Quick Test

```bash
python3 aether.py -r best64.rule --max-test-rules 100
```

### Full Benchmark + Optimize

```bash
python3 aether.py -r rules/ --visualize --optimize --max-optimize-rules 500
```

### Multiple Rule Files

```bash
python3 aether.py -r best64.rule rockyou-30000.rule d3ad0ne.rule --visualize
```

### Precision Mode for Reject / Filter Rules

```bash
python3 aether.py -r best64.rule --identical-dicts
```

### Precision Mode with Increased Word Count

```bash
python3 aether.py -r best64.rule --identical-dicts --max-words 5000 --iterations 100
```

---

## CLI Reference

| Argument | Short | Default | Description |
|---|---|---|---|
| `--rules` | `-r` | *(required)* | One or more `.rule` files or directories |
| `--output` | `-o` | `./benchmark_results` | Output directory for all results |
| `--iterations` | `-i` | `50` | GPU iterations per rule |
| `--test-runs` | | `3` | Statistical repeat runs per rule |
| `--max-words` | | `1000` | Max test words to load |
| `--max-test-rules` | | `1000` | Max rules to benchmark per file |
| `--optimize` | | `false` | Enable optimized rule set creation |
| `--max-optimize-rules` | | `500` | Max rules for the optimized set |
| `--max-time` | | `30.0` | Max total execution time for optimized set (seconds) |
| `--visualize` | | `false` | Generate all visualization charts |
| `--dpi` | | `300` | Output image DPI |
| `--visualization-output` | | `<output>/visualizations` | Separate directory for chart images |
| `--platform` | | `0` | OpenCL platform index |
| `--device` | | `0` | OpenCL device index |
| `--identical-dicts` | | `false` | Enable identical-dicts precision mode (see below) |

---

## Identical-Dicts Precision Mode

By default, Aether benchmarks rules against a small diverse set of 16 built-in passwords (`password`, `123456`, `qwerty`, etc.). This works well for most transformation rules, but **reject and filter rules** (`!`, `/`, `(`, `)`, `<`, `>`, `_`, `=`, `%`) are sensitive to word-to-word variation — whether a rule accepts or rejects a given word depends on its length and character content, introducing measurement noise across iterations.

Pass `--identical-dicts` to eliminate this noise. In this mode, the built-in 16-word list is **tiled repeatedly** to fill the `--max-words` slot count. Every cycle is the same words in the same order, so every GPU iteration sees an identical input distribution. Accept/reject decisions become fully deterministic, and the measured time reflects the pure rule overhead rather than input variance.

```bash
# Standard precision test for reject rules
python3 aether.py -r best64.rule --identical-dicts

# Higher word count = more GPU work per timing sample = lower relative noise
python3 aether.py -r best64.rule --identical-dicts --max-words 5000 --iterations 100
```

When this mode is active, the Configuration Summary prints:

```
  Identical-dicts mode: ENABLED — 1000 words (built-ins tiled)
```

The JSON performance report also records `"identical_dicts_mode": true` in its `metadata` block, so results from a precision run are always distinguishable from a standard run when comparing or archiving reports.

---

## Output Files

For each input `<name>.rule` file, Aether produces:

| File | Description |
|---|---|
| `<name>_sorted.rule` | Rules re-ordered fastest → slowest |
| `<name>_performance_report.json` | Full per-rule metrics (time, ops/sec, CV%, etc.) |
| `<name>_radar.png` | Radar chart for top 20 rules |
| `<name>_heatmap.png` | Rule-type performance heatmap |
| `<name>_statistical.png` | Statistical summary dashboard |
| `<name>_distribution.png` | Distribution + box plots |

When `--optimize` is enabled:

| File | Description |
|---|---|
| `optimized_rules.rule` | Selected fastest rule subset |
| `optimized_rules_optimization_report.json` | Optimizer parameters and selected rule data |

### JSON Report Metadata Fields

```json
{
  "metadata": {
    "total_rules": 64,
    "test_date": "2026-04-05 14:32:00",
    "test_iterations": 50,
    "test_words_count": 1000,
    "max_word_len": 64,
    "max_rule_len": 32,
    "identical_dicts_mode": false
  }
}
```

`identical_dicts_mode` is `true` when the run was performed with `--identical-dicts`.

---

## Supported Hashcat Rules

Aether's OpenCL kernel implements the following Hashcat rule operations:

### Case & Transform
| Rule | Operation |
|---|---|
| `l` | Lowercase all characters |
| `u` | Uppercase all characters |
| `c` | Capitalize first, lowercase rest |
| `C` | Lowercase first, uppercase rest |
| `t` | Toggle case of all characters |
| `T N` | Toggle case at position N |
| `E` | Title-case (capitalize after space/hyphen/underscore) |
| `e X` | Title-case after separator X |

### Structural Modifications
| Rule | Operation |
|---|---|
| `r` | Reverse the word |
| `d` | Duplicate the word |
| `f` | Append reversed word (reflect) |
| `k` | Swap first two characters |
| `K` | Swap last two characters |
| `q` | Duplicate each character in place |
| `{N` | Rotate left by N |
| `}N` | Rotate right by N |
| `[N` | Delete first N characters |
| `]N` | Delete last N characters |
| `D N` | Delete character at position N |
| `'N` | Truncate to N characters |
| `x N M` | Extract substring from N, length M |
| `O N M` | Delete M characters starting at N |

### Append / Insert
| Rule | Operation |
|---|---|
| `$X` | Append character X |
| `^X` | Prepend character X |
| `i N X` | Insert X at position N |
| `o N X` | Overwrite position N with X |
| `z` | Duplicate first character (prepend) |
| `Z` | Duplicate last character (append) |
| `y N` | Prepend first N characters |
| `Y N` | Append last N characters |
| `p N` | Repeat word N times |
| `p` (1-char) | Append `s` (pluralise) |

### Substitution
| Rule | Operation |
|---|---|
| `s X Y` | Replace all occurrences of X with Y |
| `* N M` | Swap characters at positions N and M |
| `@ X` | Delete all occurrences of X |
| `+ N` | Increment character at position N |
| `- N` | Decrement character at position N |
| `v N X` | Insert X every N characters |

### Reject / Filter Rules
| Rule | Rejects if… |
|---|---|
| `! X` | Word contains character X |
| `/ X` | Word does not contain X |
| `( X` | First character is not X |
| `) X` | Last character is not X |
| `< N` | Word length ≥ N |
| `> N` | Word length ≤ N |
| `_ N` | Word length ≠ N |
| `= N X` | Position N is not character X |
| `% N X` | Less than N occurrences of X |
| `Q` | Reject unconditionally |

> **Tip:** Use `--identical-dicts` when benchmarking files that contain a significant proportion of reject/filter rules. The tiled input ensures these rules see a consistent, predictable word distribution on every iteration.

---

## Architecture

```
aether.py
├── Colors               # ANSI terminal colour codes
├── OPENCL_KERNEL_SOURCE # Full rule engine in OpenCL C (GPU kernel)
├── print_banner()       # Startup banner
├── VisualizationEngine  # Radar, heatmap, statistical, distribution charts
├── RulePerformanceTester
│   ├── __init__()       # OpenCL context & queue setup
│   ├── setup_test_data()# Load/tile built-in word corpus; supports --identical-dicts
│   ├── setup_visualization()
│   ├── benchmark_rule() # Core GPU timing loop (statistical)
│   ├── test_rule_file_performance()
│   ├── save_sorted_rules()
│   └── save_performance_report()  # Records identical_dicts_mode in metadata
├── RuleSetOptimizer
│   └── create_optimized_set() # Merge reports, select fastest subset
├── find_rule_files()    # Recursive .rule file discovery
└── main()               # CLI entrypoint (argparse)
```

---

## How It Works

1. **Initialization** — Aether selects an OpenCL platform/device and compiles the rule engine kernel.
2. **Test Data** — A built-in corpus of 16 representative passwords is loaded. In standard mode, up to `--max-words` unique words are used. In `--identical-dicts` mode, the 16-word list is tiled to fill the full `--max-words` count, guaranteeing a uniform input distribution.
3. **Benchmarking** — Each rule in the input file(s) is dispatched to the GPU. The kernel applies the rule to all test words in parallel. Timing is repeated `--test-runs` × `--iterations` times for statistical accuracy.
4. **Statistics** — For each rule, Aether records mean, median, std deviation, coefficient of variation (CV%), min, max, and operations per second.
5. **Sorting** — Rules are ranked fastest-first and written to a new `.rule` file.
6. **Visualization** — If `--visualize` is set, charts are rendered and saved as PNG.
7. **Optimization** — If `--optimize` is set, the `RuleSetOptimizer` reads all JSON reports and selects the highest-performing rules within the time and count budget.

---

## Performance Tips

- Use `--max-test-rules` to limit rules during initial exploration.
- Use `--test-runs 5` or higher for more stable statistics on noisy systems.
- Use `--identical-dicts` when your rule file is heavy on reject/filter rules — it removes word-variance noise and makes timing comparisons more meaningful.
- Use `--platform` / `--device` to target a specific GPU when multiple OpenCL devices are present (run `clinfo` to list available devices).
- Pass a directory path to `-r` to batch-benchmark an entire rule collection in one run.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Hashcat](https://hashcat.net/hashcat/) — the original rule specification this tool benchmarks against
- Michelson-Morley interferometry methodology — inspiration for the precision-measurement approach to performance testing
- [PyOpenCL](https://documen.tician.de/pyopencl/) — Python bindings for OpenCL
