// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "utils.h"

#define WINDOW_SIZE 2

namespace diskann {
  struct FreqWindow {
    // freq[0-WINDOW_SIZE]. freq[idx] is the sum_freq.
    uint16_t freq[WINDOW_SIZE];

    FreqWindow() {
      for (int i = 0; i < WINDOW_SIZE; i++) {
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
      freq[idx] = (uint16_t)(freq[idx] - ((uint32_t)freq[0] + freq[1] - freq[idx]));
      // equals to the sum of rest three freq
      freq[nxt_idx] = (uint16_t)((uint32_t)freq[0] + freq[1] - freq[nxt_idx]);
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
      uint16_t nxt_idx = (idx + 1) % WINDOW_SIZE;
      for (uint32_t i = 0; i < n_nodes; i++) {
        freq_window[i].move(idx, nxt_idx);
      }
      idx = nxt_idx;
    }
  };
}   // namespace diskann
