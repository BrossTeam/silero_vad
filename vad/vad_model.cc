#include "vad/vad_model.h"

#include "glog/logging.h"

#include <iostream>

VadModel::VadModel(const std::string& model_path, bool denoise, int sample_rate,
                   float threshold, float min_sil_dur, float speech_pad, int ms_chunk_size)
    : OnnxModel(model_path),
    _denoise(denoise),
    _sample_rate(sample_rate),
    threshold(threshold),
    _min_sil_dur(min_sil_dur),
    _speech_pad(speech_pad) {
    _denoiser = std::make_shared<Denoiser>();
    _resampler = std::make_shared<Resampler>();
    _sample_queue = std::make_shared<SampleQueue>();
    frame_rt_size = ms_chunk_size * (16000 / 1000);
    Reset();
}

void VadModel::Reset() {
    _h.resize(SIZE_HC);
    _c.resize(SIZE_HC);
    std::memset(_h.data(), 0.0f, SIZE_HC * sizeof(float));
    std::memset(_c.data(), 0.0f, SIZE_HC * sizeof(float));
    _on_speech = false;
    _temp_stop = 0;
    pos_iterator = 0;
    _sample_queue->Clear();
    _denoiser->Reset();
}

float VadModel::Forward(const std::vector<float>& pcm) {
    std::vector<float> input_pcm{ pcm.data(), pcm.data() + pcm.size() };

    // batch_size * num_samples
    const int64_t batch_size = 1;
    int64_t input_node_dims[2] = { batch_size, (int64_t)input_pcm.size() };
    auto input_ort = Ort::Value::CreateTensor<float>(
        memory_info_, input_pcm.data(), input_pcm.size(), input_node_dims, 2);

    const int64_t sr_node_dims[1] = { batch_size };
    std::vector<int64_t> sr = { _sample_rate };
    auto sr_ort = Ort::Value::CreateTensor<int64_t>(memory_info_, sr.data(),
        batch_size, sr_node_dims, 1);
    const int64_t hc_node_dims[3] = { 2, batch_size, 64 };
    auto h_ort = Ort::Value::CreateTensor<float>(memory_info_, _h.data(), SIZE_HC,
        hc_node_dims, 3);
    auto c_ort = Ort::Value::CreateTensor<float>(memory_info_, _c.data(), SIZE_HC,
        hc_node_dims, 3);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.emplace_back(std::move(input_ort));
    ort_inputs.emplace_back(std::move(sr_ort));
    ort_inputs.emplace_back(std::move(h_ort));
    ort_inputs.emplace_back(std::move(c_ort));

    auto ort_outputs = session_->Run(
        Ort::RunOptions{ nullptr }, input_node_names_.data(), ort_inputs.data(),
        ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

    float posterier = ort_outputs[0].GetTensorMutableData<float>()[0];
    float* hn = ort_outputs[1].GetTensorMutableData<float>();
    float* cn = ort_outputs[2].GetTensorMutableData<float>();
    _h.assign(hn, hn + SIZE_HC);
    _c.assign(cn, cn + SIZE_HC);

    return posterier;
}

float VadModel::Vad(const std::vector<float>& pcm,
    std::vector<float>* start_pos,
    std::vector<float>* stop_pos) {
    std::vector<float> in_pcm{ pcm.data(), pcm.data() + pcm.size() };
    if (_denoise) {
        std::vector<float> resampled_pcm;
        std::vector<float> denoised_pcm;
        // 0. Upsample to 48k for RnNoise
        if (_sample_rate != 48000) {
            _resampler->Resample(_sample_rate, in_pcm, 48000, &resampled_pcm);
            in_pcm = resampled_pcm;
        }
        // 1. Denoise with RnNoise
        _denoiser->Denoise(in_pcm, &denoised_pcm);
        in_pcm = denoised_pcm;
        // 2. Downsample to 16k for VAD
        _resampler->Resample(48000, in_pcm, 16000, &resampled_pcm);
        _sample_rate = 16000;
        in_pcm = resampled_pcm;
    }
    _sample_queue->AcceptWaveform(in_pcm);

    // Support 512 1024 1536 samples for 16k
    int frame_ms = 64;
    int frame_size = frame_ms * (16000 / 1000);
    int num_frames = _sample_queue->NumSamples() / frame_size;
    
    for (int i = 0; i < num_frames; i++) {
        _sample_queue->Read(frame_size, &in_pcm);

        float posterier = Forward(in_pcm);
        // 1. start
        if (posterier >= threshold) {
            _temp_stop = 0;
            if (_on_speech == false) {
                _on_speech = true;
                float start = pos_iterator;
                if (start < 0) {
                    start = 0;
                }
                start_pos->emplace_back(round(start * 1000) / 1000);
            }
        }
        // 2. stop
        if (posterier < (threshold - 0.15) && _on_speech == true) {
            if (_temp_stop == 0) {
                _temp_stop = pos_iterator;
            }
            // hangover
            if (pos_iterator - _temp_stop >= _min_sil_dur) {
                _temp_stop = 0;
                _on_speech = false;
                float stop = pos_iterator;
                stop_pos->emplace_back(round(stop * 1000) / 1000);
            }
        }
        pos_iterator += 1.0 * in_pcm.size() / _sample_rate;
    }
    return pos_iterator;
}

VadModel::VadState VadModel::VadRealtime(const std::vector<float> &pcm)
{
    std::vector<float> in_pcm{pcm.data(), pcm.data() + pcm.size()};
    if (_denoise)
    {
        std::vector<float> resampled_pcm;
        std::vector<float> denoised_pcm;
        // 0. Upsample to 48k for RnNoise
        if (_sample_rate != 48000)
        {
            _resampler->Resample(_sample_rate, in_pcm, 48000, &resampled_pcm);
            in_pcm = resampled_pcm;
        }
        // 1. Denoise with RnNoise
        _denoiser->Denoise(in_pcm, &denoised_pcm);
        in_pcm = denoised_pcm;
        // 2. Downsample to 16k for VAD
        _resampler->Resample(48000, in_pcm, 16000, &resampled_pcm);
        _sample_rate = 16000;
        in_pcm = resampled_pcm;
    }
    _sample_queue->AcceptWaveform(in_pcm);

    // Support 512 1024 1536 samples for 16k
    float posterier = 0;
    pos_iterator += 1;
    if (_sample_queue->Read(frame_rt_size, &in_pcm))
        posterier = Forward(in_pcm);
    else return VadState(ERR, 0);
    // 1. start
    if (posterier >= threshold)
    {
        if (_on_speech == false)
        {
            _on_speech = true;
            float start = pos_iterator;
            _last_start = pos_iterator;
            if (_current_pad != _speech_pad)
                _current_pad = _speech_pad;
            return VadState(START, posterier);
        }
        return VadState(ON, posterier);
    }

    // 2. stop
    if (posterier < (threshold - 0.15) && _on_speech == true)
    {
        if (_current_pad == 0)
        {
            _on_speech = false;
            _last_stop = pos_iterator;
            float stop = pos_iterator;
            return VadState(STOP, posterier);
        }
        else
        {
            _current_pad -= 1;
            return VadState(ON, posterier);
        }
    }
    if (_on_speech)
        return VadState(ON, posterier);
    else
        return VadState(OFF, posterier);
}