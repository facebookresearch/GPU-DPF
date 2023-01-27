// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#include <stdio.h>
#include "distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"

using namespace std;

int main(int argc, char *argv[]) {

  // Init
  distributed_point_functions::DpfParameters parameters;
  parameters.set_log_domain_size(64);
  parameters.mutable_value_type()->mutable_integer()->set_bitsize(64);
  std::unique_ptr<distributed_point_functions::DistributedPointFunction> dpf =
    distributed_point_functions::DistributedPointFunction::Create(parameters).value();

  if (dpf == NULL) {
    cout << "Error dpf NULL... Exiting" << endl;
    exit(0);
  }

  // Actual key generation
  // "Generates a pair of keys for a DPF that evaluates to `beta` when evaluated
  // `alpha`"
  //absl::uint128 alpha = 42;
  //absl::uint128 beta = 21;
  uint64_t alpha = 42;
  uint64_t beta = 21;
  auto keypair = dpf->GenerateKeys(alpha, beta).value();

  // Test
  int num_evaluation_points = 100;    
  std::vector<absl::uint128> evaluation_points(num_evaluation_points);
  for (int i = 0; i < num_evaluation_points; ++i) {
    evaluation_points[i] = i;
  }

  auto r1 = dpf->EvaluateAt<uint64_t>(keypair.first, 0, evaluation_points).value();
  auto r2 = dpf->EvaluateAt<uint64_t>(keypair.second, 0, evaluation_points).value();

  int failed = 0;
  for (int i = 0; i < num_evaluation_points; i++) {
    auto sum = r1[i] + r2[i];
    auto truth = i == alpha ? beta : 0;
    if (sum != truth) {
      failed = 1;
    }
    cout << "Index " << i << " w/ sum: " << sum << " (expect: " << truth << ")" << endl;
  }

  if (failed) cout << "FAIL" << endl;
  else cout << "SUCCESS" << endl;
}
