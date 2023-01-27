// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

/*
  Serial CPU dpf function based on the sqrt(n) grid trick described
  - https://www.youtube.com/watch?v=y2aVgxD7DJc
  - https://www.iacr.org/archive/eurocrypt2014/84410245/84410245.pdf
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <assert.h>
#include <random>
#include <memory>
#include <vector>
#include "dpf.h"

int main(int argc, char *argv[]) {
  test_log_n_method();
  test_sqrt_n_method();
  benchmark_log_n_method_perf();
  test_flat_codewords();
}
