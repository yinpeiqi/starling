// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "utils.h"

namespace diskann {
  struct FreqWindow {
    // freq[0-4]. freq[idx] is the sum_freq.
    uint16_t freq[4];

    FreqWindow() {
      for (int i = 0; i < 4; i++) {
        freq[i] = 0;
      }
    }

    void add(int idx) {
      freq[idx]++;
    }

    uint16_t get(int idx) {
      return freq[idx];
    }

    void move(uint16_t idx, uint16_t nxt_idx) {
      // TODO: here may encounter calculate out of u16.
      // originally freq[idx] is the sum, change it to one of the rest freq.
      freq[idx] = (uint16_t)(freq[idx] - ((uint32_t)freq[0] + freq[1] + freq[2] + freq[3] - freq[idx]));
      // equals to the sum of rest three freq
      freq[nxt_idx] = (uint16_t)((uint32_t)freq[0] + freq[1] + freq[2] + freq[3] - freq[nxt_idx]);
    }
  };

  struct FreqWindowList {
    std::vector<FreqWindow> freq_window;
    uint32_t n_nodes;
    uint16_t idx;

    FreqWindowList(int n) : n_nodes(n), idx(0) {
      for (uint32_t i = 0; i < n_nodes; i++) {
        freq_window.emplace_back();
      }
    }

    void add(int i) {
      freq_window[i].add(idx);
    }

    uint16_t get(int i) {
      return freq_window[i].get(idx);
    }

    void move() {
      uint16_t nxt_idx = (idx + 1) % 4;
      for (uint32_t i = 0; i < n_nodes; i++) {
        freq_window[i].move(idx, nxt_idx);
      }
      idx = nxt_idx;
    }
  };
}   // namespace diskann
