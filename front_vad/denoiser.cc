// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "front_vad/denoiser.h"

void Denoiser::Denoise(const std::vector<float>& in_pcm,
                       std::vector<float>* out_pcm) {
  sample_queue_->AcceptWaveform(in_pcm);
  int num_frames = sample_queue_->NumSamples() / FRAME_SIZE;
  int num_out_samples = num_frames * FRAME_SIZE;

  std::vector<float> input_pcm;
  sample_queue_->Read(num_out_samples, &input_pcm);
  out_pcm->resize(num_out_samples);

  for (int i = 0; i < num_frames; i++) {
    float* in_frames = input_pcm.data() + i * FRAME_SIZE;
    float* out_frames = out_pcm->data() + i * FRAME_SIZE;
    rnnoise_process_frame(st_.get(), out_frames, in_frames);
  }
}