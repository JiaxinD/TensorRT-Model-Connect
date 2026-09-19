/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/parakeet_tdt/runtime/pipeline.h"

#include <iostream>
#include <stdexcept>

using namespace trtmc;
using namespace trtmc::parakeet_tdt;

class FakeModule final : public ITrtModule {
  public:
    explicit FakeModule(int role) : role_(role) {}
    int calls{0};
    bool reset_seen{false};
    bool malformed{false};
    bool missing{false};
    TensorMap forward(const TensorMap& in) override {
        ++calls;
        if (missing)
            return {};
        if (role_ == 0)
            return {
                {"encoder_output", {values_, {1, 2}, malformed ? DType::kInt32 : DType::kFloat32}}};
        if (role_ == 1) {
            if (*static_cast<int*>(in.at("token_id").data) == 2)
                reset_seen = *static_cast<float*>(in.at("state_h_0").data) == 0;
            return {{"pred_output", {values_, {1, malformed ? 1 : 2}, DType::kFloat32}},
                    {"next_h_0", {values_, {1, 2}, DType::kFloat32}},
                    {"next_c_0", {values_, {1, 2}, DType::kFloat32}}};
        }
        return {{"token_logits", {malformed ? nullptr : tokens_, {1, 3}, DType::kFloat32}},
                {"duration_logits", {durations_, {1, 2}, DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 0; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    int role_;
    float values_[2]{1, 1};
    float tokens_[3]{2, 0, -1};
    float durations_[2]{0, 1};
};

class FakeTokenizer final : public ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {}; }
    std::string decode(const std::vector<int32_t>& ids) const override {
        return ids == std::vector<int32_t>{0} ? "hello" : "unexpected";
    }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

int main() {
    auto encoder = std::make_unique<FakeModule>(0);
    auto predictor = std::make_unique<FakeModule>(1);
    auto joint = std::make_unique<FakeModule>(2);
    auto* enc = encoder.get();
    auto* pred = predictor.get();
    auto* joint_ptr = joint.get();
    TdtConfig cfg;
    cfg.num_mel_bins = 2;
    cfg.mel_n_fft = 4;
    cfg.mel_win_length = 4;
    cfg.mel_hop_length = 2;
    cfg.mel_length = 8;
    cfg.encoder_seq_len = 1;
    cfg.encoder_hidden_size = 2;
    cfg.pred_hidden_size = 2;
    cfg.pred_num_layers = 1;
    cfg.blank_id = 2;
    cfg.vocab_size = 2;
    cfg.duration_values = {0, 1};
    MelFilterbank filters{{1, 1, 1, 1, 1, 1}, 3, 2};
    TdtPipeline pipeline(std::move(encoder), std::move(predictor), std::move(joint), cfg, filters,
                         std::make_shared<FakeTokenizer>());
    auto bindings = pipeline.task_bindings();
    if (bindings.size() != 1 || bindings[0].key.id != "speech_transcription")
        return 1;
    auto* task = static_cast<internal::ISpeechTranscription*>(bindings[0].implementation);
    float pcm[16]{};
    internal::SpeechTranscriptionRequest req{{{pcm, 16}, 16000, 1}, {}};
    for (int i = 0; i < 2; ++i) {
        pred->reset_seen = false;
        auto result = task->run(req, {});
        if (result.text != "hello" || result.token_ids != std::vector<int32_t>{0} ||
            !pred->reset_seen)
            return 2;
    }
    internal::ConfigEntry bad{"max_new_tokens", int64_t{0}};
    const int before = enc->calls;
    try {
        task->run(req, {&bad, 1});
        return 3;
    } catch (const std::invalid_argument&) {
    }
    if (enc->calls != before)
        return 4;
    for (auto* module : {enc, pred, joint_ptr}) {
        module->malformed = true;
        try {
            task->run(req, {});
            return 5;
        } catch (const std::runtime_error&) {
        }
        module->malformed = false;
        module->missing = true;
        try {
            task->run(req, {});
            return 6;
        } catch (const std::runtime_error&) {
        }
        module->missing = false;
    }
    // A failed request must not poison the predictor state of the next request.
    pred->reset_seen = false;
    if (task->run(req, {}).text != "hello" || !pred->reset_seen)
        return 7;
    std::cout << "semantic pipeline orchestration passed (fake engines)\n";
}
