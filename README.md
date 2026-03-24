# Aether - Hashcat Rule Performance Benchmark Tool

Aether is an advanced benchmarking and analysis tool for evaluating the performance of Hashcat rule files. It measures execution speed, efficiency, and effectiveness of rules across in-build dictionariy, and can generate optimized rule sets and detailed visualizations.

---

## Features

- Benchmark Hashcat rule performance across in-build dictionary
- Support for testing multiple rules
- Configurable iterations and test runs
- Rule set optimization based on performance constraints
- Advanced visualization generation
- GPU/OpenCL device selection
- Flexible dataset handling with optional consistency testing
- Structured output and reporting

---

## Requirements

- Python 3.8 or higher
- OpenCL-compatible device (optional but recommended)
- Required Python packages:

```bash
pip install numpy pyopencl tqdm matplotlib
```

---

## Usage

```bash
python aether.py \
  -r rules/ \
  -o results \
  --iterations 50 \
  --test-runs 3
```

---

## Arguments

### Required Arguments

| Argument            | Description |
|--------------------|------------|
| -r, --rules         | Rule files or directories containing `.rule` files |

---

### Core Arguments

| Argument            | Description |
|--------------------|------------|
| -o, --output        | Output directory for results (default: ./benchmark_results) |
| -i, --iterations    | Number of test iterations per rule (default: 50) |
| --test-runs         | Number of test runs per rule (default: 3) |
| --max-test-rules    | Maximum number of rules to benchmark (default: 1000) |

---

### Optimization Arguments

| Argument                | Description |
|------------------------|------------|
| --optimize             | Enable optimized rule set generation |
| --max-optimize-rules   | Maximum rules in optimized set (default: 500) |
| --max-time             | Maximum total execution time for optimized set (default: 30.0 seconds) |
| --identical-dicts      | Use identical dictionary sets for consistency testing |

---

### Visualization Arguments

| Argument                    | Description |
|----------------------------|------------|
| --visualize                | Enable visualization generation |
| --dpi                      | DPI for output images (default: 300) |
| --visualization-output     | Separate directory for visualization output |

---

### Device Selection Arguments

| Argument        | Description |
|----------------|------------|
| --platform     | OpenCL platform index (default: 0) |
| --device       | OpenCL device index (default: 0) |

---

## How It Works

### Rule Discovery

- Loads rule files from provided paths
- Validates and prepares rules for testing

### Benchmark Execution

- Applies rules against dictionary words
- Measures execution time across multiple iterations
- Aggregates performance metrics

### Optimization (Optional)

- Selects best-performing rules based on:
  - Execution speed
  - Time constraints
- Outputs optimized rule sets and reports

### Visualization (Optional)

- Generates charts and visual summaries
- Helps identify high-performance rules
- Outputs images for analysis and reporting

---

## Output

- Benchmark results stored in output directory
- Optional optimized rule set
- JSON optimization report
- Visualization images (if enabled)

---

## Example Workflows

### Basic Benchmark

```bash
python aether.py -r best64.rule
```

### Limited Rule Testing

```bash
python aether.py -r rules/ --max-test-rules 100
```

### Full Benchmark with Optimization and Visualization

```bash
python aether.py \
  -r rules/ \
  --visualize \
  --optimize \
  --max-optimize-rules 500
  --identical-dicts
```

---

## Limitations

- Performance depends on hardware and dataset size
- Large rule sets may increase runtime significantly
- Visualization requires additional processing time

---

## Future Improvements

- Multi-GPU benchmarking support
- Distributed benchmarking
- More advanced optimization heuristics
- Interactive visualization dashboards

---

## License

MIT License

---

## Credits

Developed for advanced Hashcat rule analysis, benchmarking, and optimization workflows.
