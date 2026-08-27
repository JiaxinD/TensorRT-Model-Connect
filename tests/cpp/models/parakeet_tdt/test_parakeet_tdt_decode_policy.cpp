/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/parakeet_tdt/tdt_config.h"

#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void test_duration_value_controls_frame_advance() {
    const auto decision = trtmc::make_tdt_greedy_decision(7, 2, {0, 1, 3}, 9);
    check(decision.emit_token, "nonblank should emit");
    check(decision.frame_advance == 3, "duration table value should advance frames");
}

void test_zero_duration_blank_forces_progress_without_emit() {
    const auto decision = trtmc::make_tdt_greedy_decision(9, 0, {0, 1, 3}, 9);
    check(!decision.emit_token, "blank should not emit");
    check(decision.frame_advance == 1, "zero-duration blank must force progress");
}

void test_zero_duration_nonblank_stays_on_frame() {
    const auto decision = trtmc::make_tdt_greedy_decision(7, 0, {0, 1, 3}, 9);
    check(decision.emit_token, "zero-duration nonblank should emit");
    check(decision.frame_advance == 0, "zero-duration nonblank must stay on the frame");
}

} // namespace

int main() {
    test_duration_value_controls_frame_advance();
    test_zero_duration_blank_forces_progress_without_emit();
    test_zero_duration_nonblank_stays_on_frame();
    if (failures) {
        std::cerr << failures << " TDT decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}
