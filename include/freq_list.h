// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "utils.h"

namespace diskann {
  struct FreqWindow {
    uint16_t freq[4];
    uint16_t sum_freq;

    FreqWindow() {
      for (int i = 0; i < 4; i++) {
        freq[i] = 0;
      }
      sum_freq = 0;
    }

    void add() {
        sum_freq++;
    }

    uint16_t get() {
        return sum_freq;
    }

    void move(uint16_t idx, uint16_t nxt_idx) {
        freq[idx] = sum_freq - (freq[0] + freq[1] + freq[2] + freq[3] - freq[idx]);
        sum_freq -= freq[nxt_idx];
        freq[nxt_idx] = 0;
    }
  };

  struct FreqWindowList {
    std::vector<FreqWindow> freq_window;
    uint16_t idx;
    uint32_t n_nodes;

    FreqWindowList(int n) : n_nodes(n), idx(0) {
      for (int i = 0; i < n_nodes; i++) {
        freq_window.emplace_back();
      }
    }

    void add(int i) {
      freq_window[i].sum_freq++;
    }

    uint16_t get(int i) {
      return freq_window[i].sum_freq;
    }

    void move() {
      uint16_t nxt_idx = (idx + 1) % 4;
      for (int i = 0; i < n_nodes; i++) {
        freq_window[i].move(idx, nxt_idx);
      }
    }
  };
}   // namespace diskann
