**Aether - Hashcat Rule Performance Benchmark**

*"Just as the Michelson-Morley experiment of 1887 sought to measure the fundamental constants of the universe, this tool measures the fundamental performance characteristics of hashcat rules with scientific precision."*

```
# **Hashcat Rule Performance Benchmark**

A scientific-grade performance analysis tool for Hashcat rules with advanced visualizations and OpenCL GPU acceleration.

## Features

- **⚡GPU Accelerated**: OpenCL-powered rule performance testing
- **Advanced Visualizations**: Radar charts, heatmaps, statistical summaries
- **Scientific Methodology**: Michelson-Morley inspired precision testing
- **Rule Optimization**: Automatic generation of optimized rule sets
- **Performance Metrics**: Execution time, operations/sec, coefficient of variation

## Quick Start

### Basic Testing

python3 aether.py -r best64.rule --visualize

Full Optimization Pipeline

python3 aether.py -r test.rule \
  --iterations 200 \
  --test-runs 20 \
  --max-test-rules 10000 \
  --optimize \
  --max-optimize-rules 1000 \
  --visualize \
  --dpi 600 \
  --identical-dicts
```

**Installation**

Install Dependencies:

```
pip install pyopencl numpy matplotlib seaborn pandas scipy```

Verify OpenCL Support:

python3 -c "import pyopencl as cl; print([d.name for d in cl.get_platforms()[0].get_devices()])"
```

**Usage Examples**

*Performance Testing Only*

```
python3 aether.py -r rules/ best64.rule --test-runs 5 --iterations 100
```

*With Optimization*

```
python3 aether.py -r rules/ --optimize --max-optimize-rules 500 --max-time 30.0
```

*High-Resolution Visualizations*

```
python3 aether.py -r rules/ --visualize --dpi 600 --output ./results/
```

**Command Line Options**

*Core Options*

```
-r, --rules: Rule files or directories (required)

-o, --output: Output directory (default: ./benchmark_results)

-i, --iterations: Test iterations per rule (default: 50)

--test-runs: Test runs per rule for statistical accuracy (default: 3)

--max-test-rules: Maximum rules to test (default: 1000)
```

*Optimization*

```
--optimize: Create optimized rule set

--max-optimize-rules: Maximum rules for optimized set (default: 500)

--max-time: Maximum total time constraint (default: 30.0s)
```

*Visualization*

```
--visualize: Generate comprehensive visualizations

--dpi: Output image DPI (default: 300)

--visualization-output: Separate directory for visualizations
```

*Advanced*

```
--identical-dicts: Use identical word sets for consistency

--platform: OpenCL platform index (default: 0)

--device: OpenCL device index (default: 0)
```

**Output Files**

*_sorted.rule: Rules sorted by performance (fastest first)

*_performance_report.json: Detailed performance metrics

*_radar.png: Performance radar charts

*_heatmap.png: Rule type performance heatmaps

*_statistical.png: Comprehensive statistical summaries

*_distribution.png: Performance distribution analysis

optimized_rules.rule: Generated optimized rule set

*_optimization_report.json: Optimization parameters and results

**Performance Metrics**

- Execution Time: Average time per rule operation

- Operations/Second: Throughput measurement

- Coefficient of Variation: Statistical consistency (lower is better)

- Performance Rank: Relative performance ranking

- Statistical Outliers: Automated outlier detection and removal

**Built-in Test Words**

The tool uses 50 carefully selected built-in test words, eliminating the need for external dictionary files while maintaining testing consistency.

**Visualization Examples**

[![concentrator-MT-25000-radar.png](https://i.postimg.cc/Gm8Yzmsf/concentrator-MT-25000-radar.png)](https://postimg.cc/K1bR8Ff7)

*Performance radar chart showing top 20 rules*

[![concentrator-MT-25000-heatmap.png](https://i.postimg.cc/1X3VFmQ4/concentrator-MT-25000-heatmap.png)](https://postimg.cc/mzKrfGhG)

*Rule type performance heatmap*

**Requirements**

- Python 3.6+

- OpenCL compatible GPU or CPU

- 4GB+ RAM recommended for large rule sets

**License**

MIT License - See LICENSE file for details.

**Website:**

https://hcrt.pages.dev/aether.static_workflow
