#pragma once
#include <vector>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void resampler(std::vector<float> & in, uint32_t sampleRate, std::vector<float>& out, uint32_t out_sampleRate);

#ifdef __cplusplus
}
#endif
