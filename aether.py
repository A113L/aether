#!/usr/bin/env python3
"""
Hashcat Rule Performance Benchmark Tool with Enhanced Accuracy and Colorized Output
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

# Embedded OpenCL kernel source (same as before)
OPENCL_KERNEL_SOURCE = """
// Helper function to convert char digit/letter to int position
unsigned int char_to_pos(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    // Return a value guaranteed to fail bounds checks
    return 0xFFFFFFFF; 
}

// Helper function to get rule length
unsigned int rule_len(__global const unsigned char* rule_ptr, unsigned int max_rule_len) {
    for (unsigned int i = 0; i < max_rule_len; i++) {
        if (rule_ptr[i] == 0) return i;
    }
    return max_rule_len;
}

__kernel void bfs_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned short* rules_in,
    __global unsigned char* result_buffer,
    const unsigned int num_words,
    const unsigned int num_rules,
    const unsigned int max_word_len,
    const unsigned int max_rule_len_padded,
    const unsigned int max_output_len_padded)
{
    unsigned int global_id = get_global_id(0);
    unsigned int word_idx = global_id / num_rules;
    unsigned int rule_idx = global_id % num_rules;

    if (word_idx >= num_words) return;

    __global const unsigned char* current_word_ptr = base_words_in + word_idx * max_word_len;
    __global const unsigned short* rule_id_ptr = rules_in + rule_idx * (max_rule_len_padded + 1); 
    __global const unsigned char* rule_ptr = (__global const unsigned char*)rules_in + rule_idx * (max_rule_len_padded + 1) * sizeof(unsigned short) + sizeof(unsigned short);

    unsigned int rule_id = rule_id_ptr[0];

    __global unsigned char* result_ptr = result_buffer + global_id * max_output_len_padded;

    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {
        if (current_word_ptr[i] == 0) {
            word_len = i;
            break;
        }
    }
    
    unsigned int out_len = 0;
    bool changed_flag = false;
    
    // Zero out the result buffer for this thread
    for(unsigned int i = 0; i < max_output_len_padded; i++) {
        result_ptr[i] = 0;
    }

    // --- Unify rule ID blocks ---
    unsigned int start_id_simple = 0;
    unsigned int end_id_simple = start_id_simple + 10; // l, u, c, C, t, r, k, :, d, f
    unsigned int start_id_TD = end_id_simple;
    unsigned int end_id_TD = start_id_TD + 2; // T, D
    unsigned int start_id_s = end_id_TD;
    unsigned int end_id_s = start_id_s + 1; // s
    unsigned int start_id_A = end_id_s;
    unsigned int end_id_A = start_id_A + 3; // ^, $, @
    unsigned int start_id_groupB = end_id_A;
    unsigned int end_id_groupB = start_id_groupB + 13; // p, {, }, [, ], x, O, i, o, ', z, Z, q
    unsigned int start_id_new = end_id_groupB;
    unsigned int end_id_new = start_id_new + 13; // K, *NM, LN, RN, +N, -N, .N, ,N, yN, YN, E, eX, 3NX
    unsigned int start_id_INSERT_EVERY = end_id_new;
    unsigned int end_id_INSERT_EVERY = start_id_INSERT_EVERY + 50; // vNX INSERT EVERY rules

    // --- SIMPLE RULES IMPLEMENTATION ---
    if (rule_id >= start_id_simple && rule_id < end_id_simple) {
        unsigned char cmd = rule_ptr[0];
        
        // Copy the word first
        for(unsigned int i = 0; i < word_len; i++) {
            result_ptr[i] = current_word_ptr[i];
        }
        out_len = word_len;
        
        if (cmd == 'l') { // Lowercase all
            for(unsigned int i = 0; i < word_len; i++) {
                unsigned char c = result_ptr[i];
                if (c >= 'A' && c <= 'Z') {
                    result_ptr[i] = c + 32;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 'u') { // Uppercase all
            for(unsigned int i = 0; i < word_len; i++) {
                unsigned char c = result_ptr[i];
                if (c >= 'a' && c <= 'z') {
                    result_ptr[i] = c - 32;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 'c') { // Capitalize first letter
            if (word_len > 0) {
                unsigned char c = result_ptr[0];
                if (c >= 'a' && c <= 'z') {
                    result_ptr[0] = c - 32;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 'C') { // Lowercase first letter
            if (word_len > 0) {
                unsigned char c = result_ptr[0];
                if (c >= 'A' && c <= 'Z') {
                    result_ptr[0] = c + 32;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 't') { // Toggle case
            for(unsigned int i = 0; i < word_len; i++) {
                unsigned char c = result_ptr[i];
                if (c >= 'a' && c <= 'z') {
                    result_ptr[i] = c - 32;
                    changed_flag = true;
                } else if (c >= 'A' && c <= 'Z') {
                    result_ptr[i] = c + 32;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 'r') { // Reverse
            for(unsigned int i = 0; i < word_len; i++) {
                result_ptr[i] = current_word_ptr[word_len - 1 - i];
            }
            changed_flag = true;
        }
        else if (cmd == 'k') { // Duplicate
            if (word_len * 2 <= max_output_len_padded) {
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[word_len + i] = current_word_ptr[i];
                }
                out_len = word_len * 2;
                changed_flag = true;
            }
        }
        else if (cmd == ':') { // Duplicate and reverse
            if (word_len * 2 <= max_output_len_padded) {
                // Duplicate
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[word_len + i] = current_word_ptr[i];
                }
                // Reverse the duplicate
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[word_len + i] = current_word_ptr[word_len - 1 - i];
                }
                out_len = word_len * 2;
                changed_flag = true;
            }
        }
        else if (cmd == 'd') { // Duplicate with space
            if (word_len * 2 + 1 <= max_output_len_padded) {
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[i] = current_word_ptr[i];
                    result_ptr[word_len + 1 + i] = current_word_ptr[i];
                }
                result_ptr[word_len] = ' ';
                out_len = word_len * 2 + 1;
                changed_flag = true;
            }
        }
        else if (cmd == 'f') { // Duplicate and reverse with space
            if (word_len * 2 + 1 <= max_output_len_padded) {
                // Copy original
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[i] = current_word_ptr[i];
                }
                // Add space
                result_ptr[word_len] = ' ';
                // Add reversed
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[word_len + 1 + i] = current_word_ptr[word_len - 1 - i];
                }
                out_len = word_len * 2 + 1;
                changed_flag = true;
            }
        }
    }
    // --- T/D RULES IMPLEMENTATION ---
    else if (rule_id >= start_id_TD && rule_id < end_id_TD) {
        unsigned char cmd = rule_ptr[0];
        
        // Copy the word first
        for(unsigned int i = 0; i < word_len; i++) {
            result_ptr[i] = current_word_ptr[i];
        }
        out_len = word_len;
        
        if (cmd == 'T') { // Toggle at position N
            unsigned int N = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
            if (N != 0xFFFFFFFF && N < word_len) {
                unsigned char c = result_ptr[N];
                if (c >= 'a' && c <= 'z') {
                    result_ptr[N] = c - 32;
                    changed_flag = true;
                } else if (c >= 'A' && c <= 'Z') {
                    result_ptr[N] = c + 32;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 'D') { // Delete at position N
            unsigned int N = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
            if (N != 0xFFFFFFFF && N < word_len) {
                for(unsigned int i = N; i < word_len - 1; i++) {
                    result_ptr[i] = result_ptr[i + 1];
                }
                out_len = word_len - 1;
                changed_flag = true;
            }
        }
    }
    // --- S RULES IMPLEMENTATION ---
    else if (rule_id >= start_id_s && rule_id < end_id_s) {
        unsigned char cmd = rule_ptr[0];
        unsigned int rule_length = rule_len(rule_ptr, max_rule_len_padded);
        
        if (rule_length >= 3) { // Need at least sXY
            unsigned char find_char = rule_ptr[1];
            unsigned char replace_char = rule_ptr[2];
            
            // Copy the word first
            for(unsigned int i = 0; i < word_len; i++) {
                result_ptr[i] = current_word_ptr[i];
                if (current_word_ptr[i] == find_char) {
                    result_ptr[i] = replace_char;
                    changed_flag = true;
                }
            }
            out_len = word_len;
        }
    }
    // --- GROUP A RULES IMPLEMENTATION ---
    else if (rule_id >= start_id_A && rule_id < end_id_A) {
        unsigned char cmd = rule_ptr[0];
        
        if (cmd == '^') { // Prepend
            unsigned char prepend_char = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? rule_ptr[1] : 0;
            if (prepend_char != 0 && word_len + 1 < max_output_len_padded) {
                result_ptr[0] = prepend_char;
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[i + 1] = current_word_ptr[i];
                }
                out_len = word_len + 1;
                changed_flag = true;
            }
        }
        else if (cmd == '$') { // Append
            unsigned char append_char = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? rule_ptr[1] : 0;
            if (append_char != 0 && word_len + 1 < max_output_len_padded) {
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[i] = current_word_ptr[i];
                }
                result_ptr[word_len] = append_char;
                out_len = word_len + 1;
                changed_flag = true;
            }
        }
        else if (cmd == '@') { // Delete all instances of X
            unsigned char delete_char = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? rule_ptr[1] : 0;
            if (delete_char != 0) {
                unsigned int out_idx = 0;
                for(unsigned int i = 0; i < word_len; i++) {
                    if (current_word_ptr[i] != delete_char) {
                        result_ptr[out_idx++] = current_word_ptr[i];
                    } else {
                        changed_flag = true;
                    }
                }
                out_len = out_idx;
            }
        }
    }
    // --- GROUP B RULES IMPLEMENTATION ---
    else if (rule_id >= start_id_groupB && rule_id < end_id_groupB) {
        unsigned char cmd = rule_ptr[0];
        unsigned int N = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
        
        // Copy the word first
        for(unsigned int i = 0; i < word_len; i++) {
            result_ptr[i] = current_word_ptr[i];
        }
        out_len = word_len;
        
        if (cmd == 'p') { // Pluralize
            if (word_len + 1 < max_output_len_padded) {
                result_ptr[word_len] = 's';
                out_len = word_len + 1;
                changed_flag = true;
            }
        }
        else if (cmd == '{') { // Rotate left
            if (word_len > 1) {
                unsigned char first_char = result_ptr[0];
                for(unsigned int i = 0; i < word_len - 1; i++) {
                    result_ptr[i] = result_ptr[i + 1];
                }
                result_ptr[word_len - 1] = first_char;
                changed_flag = true;
            }
        }
        else if (cmd == '}') { // Rotate right
            if (word_len > 1) {
                unsigned char last_char = result_ptr[word_len - 1];
                for(int i = word_len - 1; i > 0; i--) {
                    result_ptr[i] = result_ptr[i - 1];
                }
                result_ptr[0] = last_char;
                changed_flag = true;
            }
        }
        else if (cmd == '[') { // Delete first character
            if (word_len > 1) {
                for(unsigned int i = 0; i < word_len - 1; i++) {
                    result_ptr[i] = current_word_ptr[i + 1];
                }
                out_len = word_len - 1;
                changed_flag = true;
            }
        }
        else if (cmd == ']') { // Delete last character
            if (word_len > 1) {
                out_len = word_len - 1;
                changed_flag = true;
            }
        }
        else if (cmd == 'x') { // Extract range N-M
            unsigned int M = (rule_len(rule_ptr, max_rule_len_padded) > 2) ? char_to_pos(rule_ptr[2]) : 0xFFFFFFFF;
            if (N != 0xFFFFFFFF && M != 0xFFFFFFFF && N <= M && M < word_len) {
                unsigned int out_idx = 0;
                for(unsigned int i = N; i <= M; i++) {
                    result_ptr[out_idx++] = current_word_ptr[i];
                }
                out_len = out_idx;
                changed_flag = true;
            }
        }
        else if (cmd == 'O') { // Overstrike at position N
            unsigned char overstrike_char = (rule_len(rule_ptr, max_rule_len_padded) > 2) ? rule_ptr[2] : 0;
            if (N != 0xFFFFFFFF && overstrike_char != 0 && N < word_len) {
                result_ptr[N] = overstrike_char;
                changed_flag = true;
            }
        }
        else if (cmd == 'i') { // Insert at position N
            unsigned char insert_char = (rule_len(rule_ptr, max_rule_len_padded) > 2) ? rule_ptr[2] : 0;
            if (N != 0xFFFFFFFF && insert_char != 0 && word_len + 1 < max_output_len_padded && N <= word_len) {
                // Shift characters right
                for(int i = word_len; i > N; i--) {
                    result_ptr[i] = result_ptr[i - 1];
                }
                result_ptr[N] = insert_char;
                out_len = word_len + 1;
                changed_flag = true;
            }
        }
        else if (cmd == 'o') { // Overwrite at position N
            unsigned char overwrite_char = (rule_len(rule_ptr, max_rule_len_padded) > 2) ? rule_ptr[2] : 0;
            if (N != 0xFFFFFFFF && overwrite_char != 0 && N < word_len) {
                result_ptr[N] = overwrite_char;
                changed_flag = true;
            }
        }
        else if (cmd == '\\'') { // Increment at position N
            if (N != 0xFFFFFFFF && N < word_len) {
                result_ptr[N] = current_word_ptr[N] + 1;
                changed_flag = true;
            }
        }
        else if (cmd == 'z') { // Duplicate first character
            if (word_len + 1 < max_output_len_padded) {
                // Shift right
                for(int i = word_len; i > 0; i--) {
                    result_ptr[i] = result_ptr[i - 1];
                }
                result_ptr[0] = current_word_ptr[0];
                out_len = word_len + 1;
                changed_flag = true;
            }
        }
        else if (cmd == 'Z') { // Duplicate last character
            if (word_len + 1 < max_output_len_padded) {
                result_ptr[word_len] = current_word_ptr[word_len - 1];
                out_len = word_len + 1;
                changed_flag = true;
            }
        }
        else if (cmd == 'q') { // Duplicate all characters
            if (word_len * 2 < max_output_len_padded) {
                unsigned int out_idx = 0;
                for(unsigned int i = 0; i < word_len; i++) {
                    result_ptr[out_idx++] = current_word_ptr[i];
                    result_ptr[out_idx++] = current_word_ptr[i];
                }
                out_len = word_len * 2;
                changed_flag = true;
            }
        }
    }
    // --- NEW RULES IMPLEMENTATION ---
    else if (rule_id >= start_id_new && rule_id < end_id_new) {
        // Copy the word first
        for(unsigned int i = 0; i < word_len; i++) {
            result_ptr[i] = current_word_ptr[i];
        }
        out_len = word_len;
        
        unsigned char cmd = rule_ptr[0];
        unsigned int N = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
        unsigned int M = (rule_len(rule_ptr, max_rule_len_padded) > 2) ? char_to_pos(rule_ptr[2]) : 0xFFFFFFFF;
        unsigned char X = (rule_len(rule_ptr, max_rule_len_padded) > 2) ? rule_ptr[2] : 0;
        unsigned char separator = (rule_len(rule_ptr, max_rule_len_padded) > 1) ? rule_ptr[1] : 0;

        if (cmd == 'K') { // 'K' (Swap last two characters)
            if (word_len >= 2) {
                result_ptr[word_len - 1] = current_word_ptr[word_len - 2];
                result_ptr[word_len - 2] = current_word_ptr[word_len - 1];
                changed_flag = true;
            }
        }
        else if (cmd == '*') { // '*NM' (Swap character at position N with character at position M)
            if (N != 0xFFFFFFFF && M != 0xFFFFFFFF && N < word_len && M < word_len && N != M) {
                unsigned char temp = result_ptr[N];
                result_ptr[N] = result_ptr[M];
                result_ptr[M] = temp;
                changed_flag = true;
            }
        }
        else if (cmd == 'L') { // 'LN' (Bitwise shift left character @ N)
            if (N != 0xFFFFFFFF && N < word_len) {
                result_ptr[N] = current_word_ptr[N] << 1;
                changed_flag = true;
            }
        }
        else if (cmd == 'R') { // 'RN' (Bitwise shift right character @ N)
            if (N != 0xFFFFFFFF && N < word_len) {
                result_ptr[N] = current_word_ptr[N] >> 1;
                changed_flag = true;
            }
        }
        else if (cmd == '+') { // '+N' (ASCII increment character @ N by 1)
            if (N != 0xFFFFFFFF && N < word_len) {
                result_ptr[N] = current_word_ptr[N] + 1;
                changed_flag = true;
            }
        }
        else if (cmd == '-') { // '-N' (ASCII decrement character @ N by 1)
            if (N != 0xFFFFFFFF && N < word_len) {
                result_ptr[N] = current_word_ptr[N] - 1;
                changed_flag = true;
            }
        }
        else if (cmd == '.') { // '.N' (Replace character @ N with value at @ N plus 1)
            if (N != 0xFFFFFFFF && N + 1 < word_len) {
                result_ptr[N] = current_word_ptr[N + 1];
                changed_flag = true;
            }
        }
        else if (cmd == ',') { // ',N' (Replace character @ N with value at @ N minus 1)
            if (N != 0xFFFFFFFF && N > 0 && N < word_len) {
                result_ptr[N] = current_word_ptr[N - 1];
                changed_flag = true;
            }
        }
        else if (cmd == 'y') { // 'yN' (Duplicate first N characters)
            if (N != 0xFFFFFFFF && N > 0 && N <= word_len) {
                unsigned int total_len = word_len + N;
                if (total_len < max_output_len_padded) {
                    // Shift original word right by N positions
                    for (int i = word_len - 1; i >= 0; i--) {
                        result_ptr[i + N] = result_ptr[i];
                    }
                    // Duplicate first N characters at the beginning
                    for (unsigned int i = 0; i < N; i++) {
                        result_ptr[i] = current_word_ptr[i];
                    }
                    out_len = total_len;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 'Y') { // 'YN' (Duplicate last N characters)
            if (N != 0xFFFFFFFF && N > 0 && N <= word_len) {
                unsigned int total_len = word_len + N;
                if (total_len < max_output_len_padded) {
                    // Append last N characters
                    for (unsigned int i = 0; i < N; i++) {
                        result_ptr[word_len + i] = current_word_ptr[word_len - N + i];
                    }
                    out_len = total_len;
                    changed_flag = true;
                }
            }
        }
        else if (cmd == 'E') { // 'E' (Title case)
            // First lowercase everything
            for (unsigned int i = 0; i < word_len; i++) {
                unsigned char c = current_word_ptr[i];
                if (c >= 'A' && c <= 'Z') {
                    result_ptr[i] = c + 32;
                } else {
                    result_ptr[i] = c;
                }
            }
            
            // Then uppercase first letter and letters after spaces
            bool capitalize_next = true;
            for (unsigned int i = 0; i < word_len; i++) {
                if (capitalize_next && result_ptr[i] >= 'a' && result_ptr[i] <= 'z') {
                    result_ptr[i] = result_ptr[i] - 32;
                    changed_flag = true;
                }
                capitalize_next = (result_ptr[i] == ' ');
            }
            out_len = word_len;
        }
        else if (cmd == 'e') { // 'eX' (Title case with custom separator)
            // First lowercase everything
            for (unsigned int i = 0; i < word_len; i++) {
                unsigned char c = current_word_ptr[i];
                if (c >= 'A' && c <= 'Z') {
                    result_ptr[i] = c + 32;
                } else {
                    result_ptr[i] = c;
                }
            }
            
            // Then uppercase first letter and letters after custom separator
            bool capitalize_next = true;
            for (unsigned int i = 0; i < word_len; i++) {
                if (capitalize_next && result_ptr[i] >= 'a' && result_ptr[i] <= 'z') {
                    result_ptr[i] = result_ptr[i] - 32;
                    changed_flag = true;
                }
                capitalize_next = (result_ptr[i] == separator);
            }
            out_len = word_len;
        }
        else if (cmd == '3') { // '3NX' (Toggle case after Nth instance of separator char)
            unsigned int separator_count = 0;
            unsigned int target_count = N;
            unsigned char sep_char = X;
            
            if (target_count != 0xFFFFFFFF) {
                for (unsigned int i = 0; i < word_len; i++) {
                    if (current_word_ptr[i] == sep_char) {
                        separator_count++;
                        if (separator_count == target_count && i + 1 < word_len) {
                            // Toggle the case of the character after the separator
                            unsigned char c = current_word_ptr[i + 1];
                            if (c >= 'a' && c <= 'z') {
                                result_ptr[i + 1] = c - 32;
                                changed_flag = true;
                            } else if (c >= 'A' && c <= 'Z') {
                                result_ptr[i + 1] = c + 32;
                                changed_flag = true;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    // --- INSERT EVERY RULES IMPLEMENTATION (vNX format) ---
    else if (rule_id >= start_id_INSERT_EVERY && rule_id < end_id_INSERT_EVERY) {
        unsigned char cmd = rule_ptr[0]; // Should be 'v'
        unsigned int rule_length = rule_len(rule_ptr, max_rule_len_padded);
        
        if (rule_length >= 3) { // Need at least vNX
            // Parse N (bytes between insertions) and X (byte to insert)
            unsigned int N = char_to_pos(rule_ptr[1]); // Number of bytes between insertions
            unsigned char X = rule_ptr[2]; // Character to insert
            
            if (N != 0xFFFFFFFF) {
                // Calculate maximum possible output length
                unsigned int insert_count = 0;
                
                // Count how many insertions we'll make
                if (N > 0) {
                    insert_count = (word_len - 1) / N; // Insert every N characters
                } else {
                    // If N=0, insert after every character
                    insert_count = word_len;
                }
                
                unsigned int max_possible_len = word_len + insert_count;
                
                if (max_possible_len < max_output_len_padded) {
                    unsigned int out_idx = 0;
                    unsigned int char_counter = 0;
                    
                    if (N == 0) {
                        // Special case: N=0 means insert after every character
                        for (unsigned int i = 0; i < word_len; i++) {
                            result_ptr[out_idx++] = current_word_ptr[i];
                            result_ptr[out_idx++] = X;
                        }
                        out_len = out_idx;
                        changed_flag = true;
                    } else {
                        // Normal case: insert every N characters
                        for (unsigned int i = 0; i < word_len; i++) {
                            result_ptr[out_idx++] = current_word_ptr[i];
                            char_counter++;
                            
                            // Insert character after every N bytes
                            if (char_counter >= N && i < word_len - 1) {
                                result_ptr[out_idx++] = X;
                                char_counter = 0; // Reset counter
                            }
                        }
                        out_len = out_idx;
                        changed_flag = true;
                    }
                }
            }
        } else if (rule_length == 2) {
            // Handle vX format (assume N=1, insert after every character)
            unsigned char X = rule_ptr[1]; // Character to insert
            
            unsigned int max_possible_len = word_len * 2;
            if (max_possible_len < max_output_len_padded) {
                unsigned int out_idx = 0;
                
                for (unsigned int i = 0; i < word_len; i++) {
                    result_ptr[out_idx++] = current_word_ptr[i];
                    result_ptr[out_idx++] = X;
                }
                
                out_len = out_idx;
                changed_flag = true;
            }
        }
    }
    
    // Final output processing
    if (changed_flag && out_len > 0) {
        if (out_len < max_output_len_padded) {
            result_ptr[out_len] = 0; // Null terminator
        }
    } else {
        // If the word was not changed or rule execution failed/resulted in length 0, zero out the output
        for (unsigned int i = 0; i < max_output_len_padded; i++) {
            result_ptr[i] = 0;
        }
    }
}
"""

class RulePerformanceTester:
    def __init__(self, platform_index=0, device_index=0):
        """Initialize OpenCL context and compile kernel"""
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
                devices = platform.get_devices(cl.device_type.ALL)
            if not devices:
                raise RuntimeError("No OpenCL devices found")
            
            device = devices[device_index]
            print(f"{Colors.GREEN}Using device: {device.name}{Colors.END}")
            print(f"{Colors.CYAN}Device memory: {device.global_mem_size // (1024*1024)} MB{Colors.END}")
            
            # Create context and queue
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context)
            
            # Build program
            self.program = cl.Program(self.context, OPENCL_KERNEL_SOURCE).build()
            
        except Exception as e:
            print(f"{Colors.RED}OpenCL initialization failed: {e}{Colors.END}")
            raise
    
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
        self.max_rule_len_padded = 32
        self.max_output_len_padded = 128
        self.iterations = 50  # Number of iterations for averaging
        
        print(f"{Colors.GREEN}Loaded {len(self.test_words)} test words{Colors.END}")
        print(f"{Colors.CYAN}Max word length: {self.max_word_len}{Colors.END}")
        print(f"{Colors.CYAN}Max rule length: {self.max_rule_len_padded}{Colors.END}")
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
        # Convert rules to the format expected by the kernel
        # Each rule is stored as: [rule_id (ushort), rule_chars...]
        rule_buffer_size = len(rules) * (self.max_rule_len_padded + 1) * 2  # *2 for ushort
        rule_data = np.zeros(rule_buffer_size // 2, dtype=np.uint16)  # ushort array
        
        # Assign rule IDs based on rule type (simplified mapping)
        for i, rule in enumerate(rules):
            rule_start = i * (self.max_rule_len_padded + 1)
            rule_data[rule_start] = self.classify_rule(rule)  # rule_id
            
            # Copy rule characters
            for j, char in enumerate(rule[:self.max_rule_len_padded]):
                rule_data[rule_start + 1 + j] = char
        
        # Create OpenCL buffer
        rules_buf = cl.Buffer(self.context, 
                            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=rule_data)
        
        return rules_buf, len(rules)
    
    def classify_rule(self, rule: bytes) -> int:
        """Classify rule type for kernel rule_id (simplified mapping)"""
        if not rule:
            return 0
        
        first_char = rule[0]
        
        # Simple rules (l, u, c, C, t, r, k, :, d, f)
        if first_char in b'lucCtrk:df':
            return 0
        # T/D rules
        elif first_char in b'TD':
            return 10
        # s rule
        elif first_char == b's':
            return 12
        # Group A rules (^, $, @)
        elif first_char in b'^$@':
            return 13
        # Group B rules
        elif first_char in b'p{}[]xOio\'zZq':
            return 16
        # New rules
        elif first_char in b'K*LR+-.,yYe3':
            return 29
        # Insert every rules (v)
        elif first_char == b'v':
            return 42
        else:
            return 0  # Default
    
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
                result_size = num_words * num_rules * self.max_output_len_padded
                result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result_size)
                
                # Set up kernel execution
                global_size = (num_words * num_rules,)
                
                # Warm-up run (discarded)
                self.program.bfs_kernel(
                    self.queue, global_size, None,
                    words_buf, rules_buf, result_buf,
                    np.uint32(num_words), np.uint32(num_rules),
                    np.uint32(self.max_word_len), np.uint32(self.max_rule_len_padded),
                    np.uint32(self.max_output_len_padded)
                )
                self.queue.finish()
                
                # Timed execution
                start_time = time.time()
                for _ in range(self.iterations):
                    self.program.bfs_kernel(
                        self.queue, global_size, None,
                        words_buf, rules_buf, result_buf,
                        np.uint32(num_words), np.uint32(num_rules),
                        np.uint32(self.max_word_len), np.uint32(self.max_rule_len_padded),
                        np.uint32(self.max_output_len_padded)
                    )
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
    
    def test_rule_file_performance(self, rule_file_path: str, test_runs: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        """Test all rules in a rule file and return sorted by performance"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}Testing rule file: {rule_file_path}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        
        # Read rules from file
        rules = self.read_rules_from_file(rule_file_path)
        if not rules:
            print(f"{Colors.RED}No rules found in {rule_file_path}{Colors.END}")
            return []
        
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
                    "max_rule_len": self.max_rule_len_padded
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
    
    def validate_rule_consistency(self, rule_performance_dict1: Dict, rule_performance_dict2: Dict, threshold_percent: float = 20.0) -> Dict[str, Any]:
        """Validate that rule performance is consistent across test runs"""
        consistent_rules = 0
        total_rules = 0
        consistency_data = []
        
        common_rules = set(rule_performance_dict1.keys()) & set(rule_performance_dict2.keys())
        
        for rule_name in common_rules:
            time1 = rule_performance_dict1[rule_name]['execution_time']
            time2 = rule_performance_dict2[rule_name]['execution_time']
            
            percent_diff = abs(time1 - time2) / min(time1, time2) * 100 if min(time1, time2) > 0 else 100
            
            is_consistent = percent_diff <= threshold_percent
            if is_consistent:
                consistent_rules += 1
            
            consistency_data.append({
                'rule': rule_name,
                'time1': time1,
                'time2': time2,
                'percent_diff': percent_diff,
                'consistent': is_consistent
            })
            
            total_rules += 1
        
        consistency_ratio = consistent_rules / total_rules if total_rules > 0 else 0
        
        return {
            'total_common_rules': total_rules,
            'consistent_rules': consistent_rules,
            'consistency_ratio': consistency_ratio,
            'threshold_percent': threshold_percent,
            'consistency_data': consistency_data
        }

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
    """Print colorful banner"""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
╔════════════════════════════════════════════════════════════════╗
║                HASHCAT RULE PERFORMANCE BENCHMARK             ║
║                  Enhanced Accuracy + Color Output             ║
╚════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)

def main():
    """Main function to run rule performance testing"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Hashcat Rule Performance Benchmark Tool with Enhanced Accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
{Colors.BOLD}Examples:{Colors.END}
  {Colors.CYAN}# Test specific rule files with specific dictionaries{Colors.END}
  python3 rule_benchmark.py -r best64.rule combinator.rule -d rockyou.txt passwords.txt
  
  {Colors.CYAN}# Test all rules in directories with optimization{Colors.END}
  python3 rule_benchmark.py -r ./rules/ -d ./dictionaries/ --optimize
  
  {Colors.CYAN}# Quick test with default settings{Colors.END}
  python3 rule_benchmark.py -r best64.rule -d rockyou.txt
  
  {Colors.CYAN}# High accuracy testing with more runs{Colors.END}
  python3 rule_benchmark.py -r best64.rule -d rockyou.txt --test-runs 10 --iterations 100
        '''
    )
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
    parser.add_argument('--optimize', action='store_true',
                       help='Create optimized rule set after benchmarking')
    parser.add_argument('--max-rules', type=int, default=500,
                       help='Maximum rules for optimized set (default: 500)')
    parser.add_argument('--max-time', type=float, default=30.0,
                       help='Maximum total time for optimized set (default: 30.0)')
    parser.add_argument('--identical-dicts', action='store_true',
                       help='Use identical dictionary sets for consistency testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find all rule files
    rule_files = find_rule_files(args.rules)
    if not rule_files:
        print(f"{Colors.RED}No rule files found in specified paths!{Colors.END}")
        return
    
    print(f"\n{Colors.GREEN}Found {len(rule_files)} rule files to test:{Colors.END}")
    for rf in rule_files:
        print(f"  {Colors.CYAN}{rf}{Colors.END}")
    
    # Create tester instance
    try:
        tester = RulePerformanceTester()
        tester.iterations = args.iterations
        tester.setup_test_data(args.dict, args.max_words, use_identical_sets=args.identical_dicts)
    except Exception as e:
        print(f"{Colors.RED}Failed to initialize tester: {e}{Colors.END}")
        return
    
    # Test each rule file
    performance_reports = []
    
    for rule_file in rule_files:
        # Test rules and get performance data
        rule_performance = tester.test_rule_file_performance(rule_file, test_runs=args.test_runs)
        
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
        
        optimizer = RuleSetOptimizer(max_rules=args.max_rules, max_total_time=args.max_time)
        optimized_file = os.path.join(args.output, "optimized_rules.rule")
        optimizer.create_optimized_set(performance_reports, optimized_file)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}Rule performance testing completed!{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}All results saved to: {os.path.abspath(args.output)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")

if __name__ == "__main__":
    main()
