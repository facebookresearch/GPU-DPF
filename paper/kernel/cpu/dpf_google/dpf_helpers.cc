// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include "dpf_helpers.h"
#include "distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"

void *DPFInitialize(int log_domain_size, int bitsize) {
  distributed_point_functions::DpfParameters parameters;
  parameters.set_log_domain_size(log_domain_size);
  parameters.mutable_value_type()->mutable_integer()->set_bitsize(bitsize);
  std::unique_ptr<distributed_point_functions::DistributedPointFunction> dpf =
    distributed_point_functions::DistributedPointFunction::Create(parameters).value();
  if (dpf == NULL) {
    std::cout << "Error dpf NULL... Exiting" << std::endl;
    exit(0);
  }
  
  return (void *)dpf.release();
}

void DPFGetKey(void *dpf_ptr, int alpha, int beta, void **k1, void **k2) {
  distributed_point_functions::DistributedPointFunction *dpf = (distributed_point_functions::DistributedPointFunction *)dpf_ptr;
  auto keypair = dpf->GenerateKeys((uint32_t)alpha, (uint32_t)beta).value();
  *k1 = keypair.first.New();
  *k2 = keypair.second.New();
  keypair.first.Swap((distributed_point_functions::DpfKey *)*k1);
  keypair.second.Swap((distributed_point_functions::DpfKey *)*k2);  
}

void DPFExpand(void *dpf_ptr, void *key_data, int N, uint32_t *out) {
  distributed_point_functions::DistributedPointFunction *dpf = (distributed_point_functions::DistributedPointFunction *)dpf_ptr;
  distributed_point_functions::DpfKey key = *((distributed_point_functions::DpfKey *)key_data);  
  std::vector<absl::uint128> evaluation_points(N);
  for (int i = 0; i < N; i++) evaluation_points[i] = i;
  std::vector<uint32_t> expanded = dpf->EvaluateAt<uint32_t>(key, 0, evaluation_points).value();
  //memcpy(out, expanded.data(), sizeof(uint32_t)*N);
}
