#!/usr/bin/env python3
"""
Hashcat Rule Performance Benchmark Tool with Advanced Visualizations
"""

import pyopencl as cl
import numpy as np
import time
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Visualization styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Simplified OpenCL kernel source for testing
OPENCL_KERNEL_SOURCE = """
__kernel void rule_processor(
    __global const uchar* words,
    __global const uchar* rules,
    __global uchar* results,
    const uint num_words,
    const uint num_rules,
    const uint max_word_len,
    const uint max_rule_len,
    const uint max_result_len)
{
    uint global_id = get_global_id(0);
    uint word_idx = global_id / num_rules;
    uint rule_idx = global_id % num_rules;

    if (word_idx >= num_words || rule_idx >= num_rules) return;

    // Calculate offsets
    uint word_offset = word_idx * max_word_len;
    uint rule_offset = rule_idx * max_rule_len;
    uint result_offset = global_id * max_result_len;

    // Get current word and rule
    __global const uchar* word = words + word_offset;
    __global const uchar* rule = rules + rule_offset;
    __global uchar* result = results + result_offset;

    // Simple rule processing for performance testing
    uint word_len = 0;
    for (uint i = 0; i < max_word_len; i++) {
        if (word[i] == 0) {
            word_len = i;
            break;
        }
    }

    uint rule_len = 0;
    for (uint i = 0; i < max_rule_len; i++) {
        if (rule[i] == 0) {
            rule_len = i;
            break;
        }
    }

    // Clear result buffer
    for (uint i = 0; i < max_result_len; i++) {
        result[i] = 0;
    }

    // Process different rule types with timing measurements
    if (rule_len > 0) {
        uchar cmd = rule[0];
        
        // Copy original word
        for (uint i = 0; i < word_len; i++) {
            result[i] = word[i];
        }
        uint out_len = word_len;

        // Apply rule transformations
        if (cmd == 'l') {
            // Lowercase all letters
            for (uint i = 0; i < word_len; i++) {
                if (result[i] >= 'A' && result[i] <= 'Z') {
                    result[i] = result[i] + 32;
                }
            }
        }
        else if (cmd == 'u') {
            // Uppercase all letters
            for (uint i = 0; i < word_len; i++) {
                if (result[i] >= 'a' && result[i] <= 'z') {
                    result[i] = result[i] - 32;
                }
            }
        }
        else if (cmd == 'c') {
            // Capitalize first letter
            if (word_len > 0 && result[0] >= 'a' && result[0] <= 'z') {
                result[0] = result[0] - 32;
            }
        }
        else if (cmd == 'r') {
            // Reverse string
            for (uint i = 0; i < word_len / 2; i++) {
                uchar temp = result[i];
                result[i] = result[word_len - 1 - i];
                result[word_len - 1 - i] = temp;
            }
        }
        else if (cmd == 'd') {
            // Duplicate word
            if (word_len * 2 <= max_result_len) {
                for (uint i = 0; i < word_len; i++) {
                    result[word_len + i] = word[i];
                }
                out_len = word_len * 2;
            }
        }
        else if (cmd == 'f') {
            // Duplicate reversed
            if (word_len * 2 <= max_result_len) {
                for (uint i = 0; i < word_len; i++) {
                    result[word_len + i] = word[word_len - 1 - i];
                }
                out_len = word_len * 2;
            }
        }
        else if (cmd == '$') {
            // Append character
            if (rule_len >= 2 && word_len + 1 < max_result_len) {
                result[word_len] = rule[1];
                out_len = word_len + 1;
            }
        }
        else if (cmd == '^') {
            // Prepend character
            if (rule_len >= 2 && word_len + 1 < max_result_len) {
                // Shift right
                for (int i = word_len; i > 0; i--) {
                    result[i] = result[i - 1];
                }
                result[0] = rule[1];
                out_len = word_len + 1;
            }
        }
        else if (cmd == '[') {
            // Delete first char
            if (word_len > 1) {
                for (uint i = 0; i < word_len - 1; i++) {
                    result[i] = result[i + 1];
                }
                out_len = word_len - 1;
            }
        }
        else if (cmd == ']') {
            // Delete last char
            if (word_len > 1) {
                out_len = word_len - 1;
            }
        }
        else if (cmd == 's') {
            // Substitute
            if (rule_len >= 3) {
                uchar find_char = rule[1];
                uchar replace_char = rule[2];
                for (uint i = 0; i < word_len; i++) {
                    if (result[i] == find_char) {
                        result[i] = replace_char;
                    }
                }
            }
        }

        // Null terminate
        if (out_len < max_result_len) {
            result[out_len] = 0;
        }
    }
}
"""

class VisualizationEngine:
    """Advanced visualization engine for benchmark results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_styles()
    
    def setup_styles(self):
        """Setup matplotlib styles for scientific visualization"""
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
    
    def create_performance_radar(self, rule_performance: List[Tuple[str, Dict]], filename: str):
        """Create radar chart showing rule performance characteristics"""
        if not rule_performance:
            return
            
        # Extract top 20 rules for readability
        top_rules = rule_performance[:20]
        rules = [f"{rule[:15]}..." if len(rule) > 15 else rule for rule, _ in top_rules]
        times = [data['execution_time'] * 1000000 for _, data in top_rules]  # Convert to microseconds
        ops_sec = [data['operations_per_sec'] / 1000 for _, data in top_rules]  # Convert to K ops/sec
        cv_values = [data['metrics']['cv_percent'] for _, data in top_rules]
        
        # Normalize for radar chart
        times_norm = self.normalize_data(times, invert=True)  # Lower time is better
        ops_norm = self.normalize_data(ops_sec)  # Higher ops/sec is better
        cv_norm = self.normalize_data(cv_values, invert=True)  # Lower CV is better
        
        categories = ['Speed\n(μs)', 'Throughput\n(K ops/sec)', 'Consistency\n(CV %)']
        
        fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (rule, time_n, ops_n, cv_n) in enumerate(zip(rules, times_norm, ops_norm, cv_norm)):
            values = [time_n, ops_n, cv_n]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=rule, markersize=6, alpha=0.7)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        
        plt.title('Rule Performance Radar Chart\n(Top 20 Rules)', size=16, pad=20)
        plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}_radar.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{Colors.GREEN}Radar chart saved: {filename}_radar.png{Colors.END}")
    
    def create_performance_heatmap(self, rule_performance: List[Tuple[str, Dict]], filename: str):
        """Create heatmap showing rule performance patterns"""
        if not rule_performance:
            return
            
        # Categorize rules by type and performance
        rule_types = {}
        for rule, data in rule_performance:
            rule_char = rule[0] if rule else '?'
            if rule_char not in rule_types:
                rule_types[rule_char] = []
            rule_types[rule_char].append(data['execution_time'] * 1000000)  # μs
        
        # Prepare heatmap data
        rule_chars = list(rule_types.keys())
        performance_data = []
        
        for char in rule_chars:
            times = rule_types[char]
            if times:  # Check if list is not empty
                avg_time = np.mean(times)
                performance_data.append(avg_time)
        
        if not performance_data:
            return
            
        # Create 2D grid for heatmap
        grid_size = int(np.ceil(np.sqrt(len(rule_chars))))
        heatmap_data = np.full((grid_size, grid_size), np.nan)
        char_labels = np.full((grid_size, grid_size), '', dtype=object)
        
        for idx, (char, perf) in enumerate(zip(rule_chars, performance_data)):
            row = idx // grid_size
            col = idx % grid_size
            if row < grid_size and col < grid_size:
                heatmap_data[row, col] = perf
                char_labels[row, col] = char
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')  # Red=slow, Green=fast
        
        # Add text annotations
        for i in range(grid_size):
            for j in range(grid_size):
                if not np.isnan(heatmap_data[i, j]):
                    text = ax.text(j, i, f"{char_labels[i, j]}\n{heatmap_data[i, j]:.1f}μs",
                                 ha="center", va="center", color="black", fontsize=8,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        plt.colorbar(im, ax=ax, label='Execution Time (μs)')
        ax.set_title('Rule Type Performance Heatmap\n(Lower = Faster)', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{Colors.GREEN}Heatmap saved: {filename}_heatmap.png{Colors.END}")
    
    def create_statistical_summary(self, rule_performance: List[Tuple[str, Dict]], filename: str):
        """Create comprehensive statistical summary visualization"""
        if not rule_performance:
            return
            
        # Extract data
        rules = [rule for rule, _ in rule_performance]
        times = [data['execution_time'] * 1000000 for _, data in rule_performance]
        cv_values = [data['metrics']['cv_percent'] for _, data in rule_performance]
        ops_sec = [data['operations_per_sec'] for _, data in rule_performance]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance distribution
        axes[0,0].hist(times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.2f}μs')
        axes[0,0].axvline(np.median(times), color='green', linestyle='--', label=f'Median: {np.median(times):.2f}μs')
        axes[0,0].set_xlabel('Execution Time (μs)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Performance Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Consistency vs Performance
        scatter = axes[0,1].scatter(times, cv_values, c=ops_sec, cmap='viridis', 
                                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0,1].set_xlabel('Execution Time (μs)')
        axes[0,1].set_ylabel('Coefficient of Variation (%)')
        axes[0,1].set_title('Performance vs Consistency')
        plt.colorbar(scatter, ax=axes[0,1], label='Operations/sec')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Top 10 fastest rules
        top_10 = rule_performance[:10]
        top_rules = [f"{rule[:15]}..." if len(rule) > 15 else rule for rule, _ in top_10]
        top_times = [data['execution_time'] * 1000000 for _, data in top_10]
        
        bars = axes[1,0].barh(top_rules, top_times, 
                            color=plt.cm.Greens_r(np.linspace(0.2, 0.8, len(top_rules))))
        axes[1,0].set_xlabel('Execution Time (μs)')
        axes[1,0].set_title('Top 10 Fastest Rules')
        axes[1,0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            axes[1,0].text(width, bar.get_y() + bar.get_height()/2, 
                         f'{width:.2f}μs', ha='left', va='center', fontsize=8)
        
        # 4. Performance categories
        fast = len([t for t in times if t < 10])
        medium = len([t for t in times if 10 <= t < 100])
        slow = len([t for t in times if t >= 100])
        
        categories = ['Fast (<10μs)', 'Medium (10-100μs)', 'Slow (≥100μs)']
        counts = [fast, medium, slow]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        axes[1,1].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Performance Categories Distribution')
        
        plt.suptitle('Comprehensive Statistical Summary', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}_statistical.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{Colors.GREEN}Statistical summary saved: {filename}_statistical.png{Colors.END}")
    
    def normalize_data(self, data: List[float], invert: bool = False) -> List[float]:
        """Normalize data to 0-1 range"""
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return [0.5] * len(data)  # All values are equal
        
        normalized = [(x - min_val) / (max_val - min_val) for x in data]
        
        if invert:
            normalized = [1 - x for x in normalized]
        
        return normalized
    
    def generate_dashboard(self, performance_data: Dict, filename: str):
        """Generate a comprehensive dashboard with all visualizations"""
        print(f"{Colors.CYAN}Generating comprehensive visualization dashboard...{Colors.END}")
        
        # Create all visualizations
        if 'rule_performance' in performance_data:
            rule_performance = performance_data['rule_performance']
            
            self.create_performance_radar(rule_performance, filename)
            self.create_performance_heatmap(rule_performance, filename)
            self.create_statistical_summary(rule_performance, filename)
        
        print(f"{Colors.GREEN}Dashboard generation complete!{Colors.END}")

class RulePerformanceTester:
    def __init__(self, platform_index=0, device_index=0):
        """Initialize OpenCL context and compile kernel"""
        self.visualizer = None
        self.setup_opencl(platform_index, device_index)
        
    def setup_opencl(self, platform_index: int, device_index: int):
        """Set up OpenCL context, queue, and program"""
        try:
            # Get platform and device
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            
            platform = platforms[platform_index]
            devices = platform.get_devices(cl.device_type.GPU)
            if not devices:
                print(f"{Colors.YELLOW}No GPU devices found, trying CPU...{Colors.END}")
                devices = platform.get_devices(cl.device_type.CPU)
            if not devices:
                raise RuntimeError("No OpenCL devices found")
            
            device = devices[device_index]
            print(f"{Colors.GREEN}Using device: {device.name}{Colors.END}")
            print(f"{Colors.CYAN}Device memory: {device.global_mem_size // (1024*1024)} MB{Colors.END}")
            
            # Create context and queue
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context)
            
            # Build program with error handling
            try:
                self.program = cl.Program(self.context, OPENCL_KERNEL_SOURCE).build()
                # Test that kernel exists
                self.kernel = cl.Kernel(self.program, 'rule_processor')
                print(f"{Colors.GREEN}OpenCL kernel compiled successfully{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}Kernel compilation failed: {e}{Colors.END}")
                # Try to get build log
                try:
                    build_log = self.program.get_build_info(device, cl.program_build_info.LOG)
                    print(f"{Colors.YELLOW}Build log: {build_log}{Colors.END}")
                except:
                    pass
                raise
            
        except Exception as e:
            print(f"{Colors.RED}OpenCL initialization failed: {e}{Colors.END}")
            raise
    
    def setup_visualization(self, output_dir: str):
        """Initialize visualization engine"""
        self.visualizer = VisualizationEngine(output_dir)
        print(f"{Colors.GREEN}Visualization engine initialized{Colors.END}")
    
    def setup_test_data(self, dictionary_paths: List[str], max_words: int = 1000, use_identical_sets: bool = True):
        """Set up test parameters and load dictionaries"""
        if use_identical_sets and len(dictionary_paths) >= 1:
            # For consistency testing, use only the first dictionary
            primary_dict = dictionary_paths[0]
            self.test_words = self.load_dictionaries([primary_dict], max_words)
            print(f"{Colors.CYAN}Using identical word set from {primary_dict} for all tests{Colors.END}")
        else:
            # Original behavior - combine dictionaries
            self.test_words = self.load_dictionaries(dictionary_paths, max_words)
        
        if not self.test_words:
            # Fallback test words if no dictionaries found
            self.test_words = [
                b"password", b"123456", b"qwerty", b"letmein", 
                b"welcome", b"monkey", b"dragon", b"master",
                b"hello", b"freedom", b"whatever", b"computer",
                b"internet", b"sunshine", b"princess", b"charlie"
            ]
            print(f"{Colors.YELLOW}Using built-in test words (no dictionaries loaded){Colors.END}")
        
        # Configuration parameters
        self.max_word_len = 64
        self.max_rule_len = 32
        self.max_result_len = 128
        self.iterations = 50  # Number of iterations for averaging
        
        print(f"{Colors.GREEN}Loaded {len(self.test_words)} test words{Colors.END}")
        print(f"{Colors.CYAN}Max word length: {self.max_word_len}{Colors.END}")
        print(f"{Colors.CYAN}Max rule length: {self.max_rule_len}{Colors.END}")
        print(f"{Colors.CYAN}Test iterations: {self.iterations}{Colors.END}")
    
    def load_dictionaries(self, dictionary_paths: List[str], max_words: int = 1000) -> List[bytes]:
        """Load words from dictionary files"""
        words = []
        total_loaded = 0
        
        for dict_path in dictionary_paths:
            if total_loaded >= max_words:
                break
                
            if os.path.isfile(dict_path):
                try:
                    with open(dict_path, 'rb') as f:
                        for line in f:
                            if total_loaded >= max_words:
                                break
                            word = line.strip()
                            if word and len(word) < self.max_word_len:
                                words.append(word)
                                total_loaded += 1
                    print(f"{Colors.GREEN}Loaded {total_loaded} words from {dict_path}{Colors.END}")
                except Exception as e:
                    print(f"{Colors.RED}Error loading dictionary {dict_path}: {e}{Colors.END}")
            elif os.path.isdir(dict_path):
                # Load all files in directory
                for file_path in Path(dict_path).glob('*'):
                    if file_path.is_file() and total_loaded < max_words:
                        try:
                            with open(file_path, 'rb') as f:
                                for line in f:
                                    if total_loaded >= max_words:
                                        break
                                    word = line.strip()
                                    if word and len(word) < self.max_word_len:
                                        words.append(word)
                                        total_loaded += 1
                            print(f"{Colors.GREEN}Loaded {total_loaded} words from {file_path}{Colors.END}")
                        except Exception as e:
                            print(f"{Colors.RED}Error loading dictionary {file_path}: {e}{Colors.END}")
        
        return words
    
    def calculate_performance_metrics(self, execution_times: List[float]) -> Dict[str, Any]:
        """Calculate robust performance metrics with outlier removal"""
        if not execution_times:
            return {}
        
        # Remove outliers (values beyond 2 standard deviations)
        mean = np.mean(execution_times)
        std = np.std(execution_times)
        filtered_times = [t for t in execution_times if abs(t - mean) <= 2 * std]
        
        if not filtered_times:
            filtered_times = execution_times
        
        cv_percent = (np.std(filtered_times) / np.mean(filtered_times)) * 100 if np.mean(filtered_times) > 0 else 0
        
        return {
            'mean_time': np.mean(filtered_times),
            'median_time': np.median(filtered_times),
            'std_time': np.std(filtered_times),
            'min_time': min(filtered_times),
            'max_time': max(filtered_times),
            'cv_percent': cv_percent,  # Coefficient of variation
            'sample_size': len(filtered_times),
            'outliers_removed': len(execution_times) - len(filtered_times)
        }
    
    def prepare_rule_buffers(self, rules: List[bytes]) -> Tuple[cl.Buffer, int]:
        """Prepare rule buffers for OpenCL kernel"""
        # Convert rules to padded format
        rule_buffer_size = len(rules) * self.max_rule_len
        rule_data = np.zeros(rule_buffer_size, dtype=np.uint8)
        
        for i, rule in enumerate(rules):
            rule_start = i * self.max_rule_len
            for j, char in enumerate(rule[:self.max_rule_len - 1]):
                rule_data[rule_start + j] = char
        
        # Create OpenCL buffer
        rules_buf = cl.Buffer(self.context, 
                            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=rule_data)
        
        return rules_buf, len(rules)
    
    def prepare_word_buffers(self, words: List[bytes]) -> Tuple[cl.Buffer, int]:
        """Prepare word buffers for OpenCL kernel"""
        # Convert words to padded format
        word_buffer_size = len(words) * self.max_word_len
        word_data = np.zeros(word_buffer_size, dtype=np.uint8)
        
        for i, word in enumerate(words):
            word_start = i * self.max_word_len
            for j, char in enumerate(word[:self.max_word_len - 1]):
                word_data[word_start + j] = char
        
        # Create OpenCL buffer
        words_buf = cl.Buffer(self.context,
                            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=word_data)
        
        return words_buf, len(words)
    
    def test_single_rule_performance(self, rule: bytes, test_runs: int = 5) -> Dict[str, Any]:
        """Test performance of a single rule with multiple test runs for accuracy"""
        try:
            execution_times = []
            
            for run in range(test_runs):
                # Prepare test data
                words_buf, num_words = self.prepare_word_buffers(self.test_words)
                rules_buf, num_rules = self.prepare_rule_buffers([rule])
                
                # Result buffer
                result_size = num_words * num_rules * self.max_result_len
                result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result_size)
                
                # Set up kernel execution
                global_size = (num_words * num_rules,)
                
                # Warm-up run (discarded)
                self.kernel.set_args(words_buf, rules_buf, result_buf,
                                   np.uint32(num_words), np.uint32(num_rules),
                                   np.uint32(self.max_word_len), np.uint32(self.max_rule_len),
                                   np.uint32(self.max_result_len))
                
                cl.enqueue_nd_range_kernel(self.queue, self.kernel, global_size, None)
                self.queue.finish()
                
                # Timed execution
                start_time = time.time()
                for _ in range(self.iterations):
                    cl.enqueue_nd_range_kernel(self.queue, self.kernel, global_size, None)
                self.queue.finish()
                end_time = time.time()
                
                # Calculate average time per iteration
                total_time = end_time - start_time
                avg_time = total_time / self.iterations
                execution_times.append(avg_time)
                
                # Cleanup
                words_buf.release()
                rules_buf.release()
                result_buf.release()
            
            # Calculate robust performance metrics
            metrics = self.calculate_performance_metrics(execution_times)
            
            # Calculate operations per second based on mean time
            total_operations = num_words * num_rules * self.iterations
            operations_per_sec = total_operations / metrics['mean_time'] if metrics['mean_time'] > 0 else 0
            
            return {
                'execution_time': metrics['mean_time'],
                'total_operations': total_operations,
                'operations_per_sec': operations_per_sec,
                'success': True,
                'metrics': metrics,
                'test_runs': test_runs
            }
            
        except Exception as e:
            print(f"{Colors.RED}Error testing rule {rule}: {e}{Colors.END}")
            return {
                'execution_time': float('inf'),
                'total_operations': 0,
                'operations_per_sec': 0,
                'success': False,
                'error': str(e)
            }
    
    def test_rule_file_performance(self, rule_file_path: str, test_runs: int = 3, max_test_rules: int = 1000) -> List[Tuple[str, Dict[str, Any]]]:
        """Test all rules in a rule file and return sorted by performance"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}Testing rule file: {rule_file_path}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        
        # Read rules from file
        rules = self.read_rules_from_file(rule_file_path)
        if not rules:
            print(f"{Colors.RED}No rules found in {rule_file_path}{Colors.END}")
            return []
        
        # Limit number of rules for testing
        if len(rules) > max_test_rules:
            print(f"{Colors.YELLOW}Limiting to first {max_test_rules} rules (out of {len(rules)}){Colors.END}")
            rules = rules[:max_test_rules]
        
        # Test each rule
        rule_performance = []
        total_rules = len(rules)
        successful_tests = 0
        
        for i, rule in enumerate(rules):
            rule_str = rule.decode('ascii', errors='ignore')
            print(f"{Colors.WHITE}  Testing rule {i+1}/{total_rules}: {Colors.YELLOW}{rule_str}{Colors.END}")
            
            performance_data = self.test_single_rule_performance(rule, test_runs)
            
            if performance_data['success']:
                successful_tests += 1
                rule_performance.append((rule_str, performance_data))
                
                # Color code based on performance
                time_color = Colors.GREEN if performance_data['execution_time'] < 0.001 else Colors.YELLOW if performance_data['execution_time'] < 0.01 else Colors.RED
                cv_color = Colors.GREEN if performance_data['metrics']['cv_percent'] < 10 else Colors.YELLOW if performance_data['metrics']['cv_percent'] < 20 else Colors.RED
                
                print(f"    {Colors.CYAN}Time:{Colors.END} {time_color}{performance_data['execution_time']:.6f}s{Colors.END} "
                      f"{Colors.CYAN}Ops/sec:{Colors.END} {Colors.MAGENTA}{performance_data['operations_per_sec']:,.0f}{Colors.END} "
                      f"{Colors.CYAN}CV:{Colors.END} {cv_color}{performance_data['metrics']['cv_percent']:.1f}%{Colors.END} "
                      f"{Colors.CYAN}Runs:{Colors.END} {performance_data['test_runs']}")
            else:
                print(f"    {Colors.RED}FAILED: {performance_data.get('error', 'Unknown error')}{Colors.END}")
        
        # Sort by execution time (fastest first)
        rule_performance.sort(key=lambda x: x[1]['execution_time'])
        
        success_color = Colors.GREEN if successful_tests == total_rules else Colors.YELLOW if successful_tests > total_rules * 0.8 else Colors.RED
        print(f"\n{success_color}Completed: {successful_tests}/{total_rules} rules successful{Colors.END}")
        
        # Generate visualizations if visualizer is available
        if self.visualizer and rule_performance:
            base_name = os.path.splitext(os.path.basename(rule_file_path))[0]
            self.visualizer.generate_dashboard({
                'rule_performance': rule_performance,
                'test_runs': test_runs,
                'timestamp': datetime.now().isoformat()
            }, base_name)
        
        return rule_performance
    
    def read_rules_from_file(self, file_path: str) -> List[bytes]:
        """Read rules from hashcat rule file"""
        rules = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:  # Only add non-empty lines
                        rules.append(line.encode('ascii', errors='ignore'))
            print(f"{Colors.GREEN}Read {len(rules)} rules from {file_path}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Error reading rule file {file_path}: {e}{Colors.END}")
        
        return rules
    
    def save_sorted_rules(self, rule_performance: List[Tuple[str, Dict[str, Any]]], output_file: str):
        """Save rules sorted by performance to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Rules sorted by performance (fastest first)\n")
                f.write(f"# Generated by RulePerformanceTester\n")
                f.write(f"# Total rules: {len(rule_performance)}\n")
                f.write(f"# Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Format: rule # execution_time_seconds operations_per_second coefficient_of_variation\n\n")
                
                for rule, perf_data in rule_performance:
                    f.write(f"{rule} # {perf_data['execution_time']:.6f}s {perf_data['operations_per_sec']:,.0f} ops/sec {perf_data['metrics']['cv_percent']:.1f}% CV\n")
            
            print(f"{Colors.GREEN}Sorted rules saved to: {output_file}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}Error saving sorted rules: {e}{Colors.END}")
    
    def save_performance_report(self, rule_performance: List[Tuple[str, Dict[str, Any]]], report_file: str):
        """Save detailed performance report as JSON"""
        try:
            report_data = {
                "metadata": {
                    "total_rules": len(rule_performance),
                    "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_iterations": self.iterations,
                    "test_words_count": len(self.test_words),
                    "max_word_len": self.max_word_len,
                    "max_rule_len": self.max_rule_len
                },
                "rules": [
                    {
                        "rule": rule,
                        "execution_time_seconds": perf_data['execution_time'],
                        "operations_per_second": perf_data['operations_per_sec'],
                        "total_operations": perf_data['total_operations'],
                        "performance_rank": i + 1,
                        "statistical_metrics": perf_data['metrics']
                    }
                    for i, (rule, perf_data) in enumerate(rule_performance)
                ]
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"{Colors.GREEN}Performance report saved to: {report_file}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}Error saving performance report: {e}{Colors.END}")

class RuleSetOptimizer:
    def __init__(self, max_rules: int = 1000, max_total_time: float = 60.0):
        self.max_rules = max_rules
        self.max_total_time = max_total_time
    
    def create_optimized_set(self, performance_reports: List[str], output_file: str):
        """Create optimized rule set from multiple performance reports"""
        all_rules = []
        
        # Load performance data from all reports
        for report_file in performance_reports:
            if os.path.exists(report_file):
                try:
                    with open(report_file, 'r') as f:
                        data = json.load(f)
                        all_rules.extend(data['rules'])
                    print(f"{Colors.GREEN}Loaded performance data from {report_file}{Colors.END}")
                except Exception as e:
                    print(f"{Colors.RED}Error loading performance report {report_file}: {e}{Colors.END}")
        
        if not all_rules:
            print(f"{Colors.RED}No performance data found!{Colors.END}")
            return
        
        # Sort by execution time (fastest first)
        all_rules.sort(key=lambda x: x['execution_time_seconds'])
        
        # Select rules based on constraints
        selected_rules = []
        total_time = 0.0
        
        for rule in all_rules:
            if (len(selected_rules) < self.max_rules and 
                total_time + rule['execution_time_seconds'] <= self.max_total_time):
                selected_rules.append(rule)
                total_time += rule['execution_time_seconds']
        
        # Save optimized rule set
        with open(output_file, 'w') as f:
            f.write("# Optimized rule set - fastest rules\n")
            f.write(f"# Total rules: {len(selected_rules)}\n")
            f.write(f"# Estimated total time: {total_time:.6f}s\n")
            f.write(f"# Max rules constraint: {self.max_rules}\n")
            f.write(f"# Max time constraint: {self.max_total_time}s\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for rule in selected_rules:
                f.write(f"{rule['rule']}\n")
        
        # Save optimization report
        report_file = output_file.replace('.rule', '_optimization_report.json')
        optimization_report = {
            "optimization_parameters": {
                "max_rules": self.max_rules,
                "max_total_time": self.max_total_time,
                "actual_rules_selected": len(selected_rules),
                "actual_total_time": total_time
            },
            "selected_rules": selected_rules
        }
        
        with open(report_file, 'w') as f:
            json.dump(optimization_report, f, indent=2)
        
        print(f"\n{Colors.GREEN}Optimized rule set created: {output_file}{Colors.END}")
        print(f"{Colors.CYAN}Selected {len(selected_rules)} rules with total time {total_time:.6f}s{Colors.END}")
        print(f"{Colors.CYAN}Optimization report: {report_file}{Colors.END}")

def find_rule_files(rule_paths: List[str]) -> List[str]:
    """Find all rule files in given paths"""
    rule_files = []
    
    for path in rule_paths:
        if os.path.isfile(path):
            if path.endswith('.rule'):
                rule_files.append(path)
            else:
                print(f"{Colors.YELLOW}Warning: {path} is not a .rule file, skipping{Colors.END}")
        elif os.path.isdir(path):
            # Find all .rule files in directory
            rule_files_found = list(Path(path).rglob('*.rule'))
            if rule_files_found:
                rule_files.extend([str(f) for f in rule_files_found])
                print(f"{Colors.GREEN}Found {len(rule_files_found)} rule files in {path}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}Warning: No .rule files found in directory {path}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}Warning: Path not found: {path}{Colors.END}")
    
    # Remove duplicates and sort
    rule_files = sorted(list(set(rule_files)))
    return rule_files

def print_banner():
    """Print enhanced banner with visualization mention"""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
╔════════════════════════════════════════════════════════════════╗
║                HASHCAT RULE PERFORMANCE BENCHMARK             ║
║               Advanced Visualization Edition                  ║
║                  Michelson-Morley Inspired                    ║
╚════════════════════════════════════════════════════════════════╝
{Colors.END}
{Colors.YELLOW}
🔬 Scientific-Grade Performance Analysis
📊 Advanced Data Visualization  
⚡ OpenCL GPU Acceleration
🎯 Michelson-Morley Precision Methodology
{Colors.END}
"""
    print(banner)

def main():
    """Main function to run rule performance testing with visualizations"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Hashcat Rule Performance Benchmark Tool with Advanced Visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
{Colors.BOLD}Examples:{Colors.END}
  {Colors.CYAN}# Basic testing with visualizations{Colors.END}
  python3 rule_benchmark.py -r best64.rule -d rockyou.txt --visualize
  
  {Colors.CYAN}# Limit number of rules for quick testing{Colors.END}
  python3 rule_benchmark.py -r best64.rule -d rockyou.txt --max-test-rules 100
  
  {Colors.CYAN}# Full testing with optimization{Colors.END}
  python3 rule_benchmark.py -r best64.rule -d rockyou.txt --visualize --optimize --max-optimize-rules 500
        '''
    )
    
    # Testing arguments
    parser.add_argument('--rules', '-r', nargs='+', required=True,
                       help='Rule files or directories containing .rule files')
    parser.add_argument('--dict', '-d', nargs='+', required=True,
                       help='Dictionary files or directories for test words')
    parser.add_argument('--output', '-o', default='./benchmark_results',
                       help='Output directory for results (default: ./benchmark_results)')
    parser.add_argument('--iterations', '-i', type=int, default=50,
                       help='Number of test iterations per rule (default: 50)')
    parser.add_argument('--test-runs', type=int, default=3,
                       help='Number of test runs per rule for statistical accuracy (default: 3)')
    parser.add_argument('--max-words', type=int, default=1000,
                       help='Maximum number of test words to load (default: 1000)')
    parser.add_argument('--max-test-rules', type=int, default=1000,
                       help='Maximum number of rules to test (default: 1000)')
    
    # Optimization arguments
    parser.add_argument('--optimize', action='store_true',
                       help='Create optimized rule set after benchmarking')
    parser.add_argument('--max-optimize-rules', type=int, default=500,
                       help='Maximum rules for optimized set (default: 500)')
    parser.add_argument('--max-time', type=float, default=30.0,
                       help='Maximum total time for optimized set (default: 30.0)')
    parser.add_argument('--identical-dicts', action='store_true',
                       help='Use identical dictionary sets for consistency testing')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Generate comprehensive visualizations')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output images (default: 300)')
    parser.add_argument('--visualization-output', default=None,
                       help='Separate output directory for visualizations')
    
    # Device selection arguments
    parser.add_argument('--platform', type=int, default=0,
                       help='OpenCL platform index (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                       help='OpenCL device index (default: 0)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    viz_output = args.visualization_output or os.path.join(args.output, 'visualizations')
    os.makedirs(viz_output, exist_ok=True)
    
    # Set matplotlib DPI
    plt.rcParams['figure.dpi'] = args.dpi
    
    # Find all rule files
    rule_files = find_rule_files(args.rules)
    if not rule_files:
        print(f"{Colors.RED}No rule files found in specified paths!{Colors.END}")
        return
    
    print(f"\n{Colors.GREEN}Found {len(rule_files)} rule files to test:{Colors.END}")
    for rf in rule_files:
        print(f"  {Colors.CYAN}{rf}{Colors.END}")
    
    # Create tester instance with proper device selection
    try:
        tester = RulePerformanceTester(platform_index=args.platform, device_index=args.device)
        tester.iterations = args.iterations
        
        # Setup visualization if requested
        if args.visualize:
            tester.setup_visualization(viz_output)
            print(f"{Colors.CYAN}Advanced visualization engine activated{Colors.END}")
        
        tester.setup_test_data(args.dict, args.max_words, use_identical_sets=args.identical_dicts)
    except Exception as e:
        print(f"{Colors.RED}Failed to initialize tester: {e}{Colors.END}")
        return
    
    # Test each rule file
    performance_reports = []
    
    for rule_file in rule_files:
        # Test rules and get performance data
        rule_performance = tester.test_rule_file_performance(rule_file, test_runs=args.test_runs, max_test_rules=args.max_test_rules)
        
        if rule_performance:
            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(rule_file))[0]
            sorted_rules_file = os.path.join(args.output, f"{base_name}_sorted.rule")
            report_file = os.path.join(args.output, f"{base_name}_performance_report.json")
            
            # Save sorted rules
            tester.save_sorted_rules(rule_performance, sorted_rules_file)
            
            # Save detailed report
            tester.save_performance_report(rule_performance, report_file)
            performance_reports.append(report_file)
            
            # Print summary with colors
            print(f"\n{Colors.BOLD}{Colors.CYAN}Performance Summary for {rule_file}:{Colors.END}")
            fastest_time = rule_performance[0][1]['execution_time']
            fastest_color = Colors.GREEN if fastest_time < 0.001 else Colors.YELLOW if fastest_time < 0.01 else Colors.RED
            print(f"  {Colors.WHITE}Fastest rule:{Colors.END} {Colors.YELLOW}{rule_performance[0][0]}{Colors.END} {Colors.CYAN}({fastest_color}{rule_performance[0][1]['execution_time']:.6f}s{Colors.CYAN}){Colors.END}")
            
            slowest_time = rule_performance[-1][1]['execution_time']
            slowest_color = Colors.GREEN if slowest_time < 0.001 else Colors.YELLOW if slowest_time < 0.01 else Colors.RED
            print(f"  {Colors.WHITE}Slowest rule:{Colors.END} {Colors.YELLOW}{rule_performance[-1][0]}{Colors.END} {Colors.CYAN}({slowest_color}{rule_performance[-1][1]['execution_time']:.6f}s{Colors.CYAN}){Colors.END}")
            
            avg_time = sum(p[1]['execution_time'] for p in rule_performance) / len(rule_performance)
            avg_color = Colors.GREEN if avg_time < 0.001 else Colors.YELLOW if avg_time < 0.01 else Colors.RED
            print(f"  {Colors.WHITE}Average time:{Colors.END} {avg_color}{avg_time:.6f}s{Colors.END}")
            print()
    
    # Create optimized rule set if requested
    if args.optimize and performance_reports:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}Creating Optimized Rule Set{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        
        optimizer = RuleSetOptimizer(max_rules=args.max_optimize_rules, max_total_time=args.max_time)
        optimized_file = os.path.join(args.output, "optimized_rules.rule")
        optimizer.create_optimized_set(performance_reports, optimized_file)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}Rule performance testing completed!{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}All results saved to: {os.path.abspath(args.output)}{Colors.END}")
    if args.visualize:
        print(f"{Colors.BOLD}{Colors.MAGENTA}Visualizations saved to: {os.path.abspath(viz_output)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")

if __name__ == "__main__":
    main()
