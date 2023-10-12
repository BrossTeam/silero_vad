#ifndef VAD_VAD_MODEL_H_
#define VAD_VAD_MODEL_H_

#include "vad/onnx_model.h"

#include <memory>

#include "front_vad/denoiser.h"
#include "front_vad/resampler.h"
#include "front_vad/sample_queue.h"

#define SIZE_HC 128  // 2 * 1 * 64

class VadModel : public OnnxModel {
public:
    // types
    enum EnumVadState // the current state of an audio chunk
    {
        START, // the first block of voice
        ON,    // an ongoing block of voice
        STOP,  // the first non-voice block
        OFF,   // ongoing non-voice block
        ERR    // there was some issue with the audio chunk
    };
    struct VadState
    {
        EnumVadState state;
        float intensity;
        VadState(EnumVadState state_, float intensity_)
        {
            state = state_;
            intensity = intensity_;
        }
    };

    // vars
    std::vector<float> *start_rt_pos, *stop_rt_pos;
    int frame_rt_size;
    int pos_iterator = 0;
    float threshold;

    // funcs
    VadModel(const std::string &model_path = "data/silero_vad.onnx", bool denoise = true, int sample_rate = 16000,
             float threshold = 0.5, float min_sil_dur = 0, float speech_pad = 0, int ms_chunk_size = 32);

    void Reset();
    float Vad(const std::vector<float>& pcm, std::vector<float>* start_pos,
        std::vector<float>* stop_pos);
    VadState VadRealtime(const std::vector<float> &pcm);
    float Forward(const std::vector<float>& pcm);

private:

    bool _denoise = false;
    int _sample_rate;
    float _min_sil_dur;
    float _speech_pad;
    int _last_start;
    int _last_stop;
    int _current_pad;

    // model states
    bool _on_speech = false;
    float _temp_stop = 0;

    // Onnx model
    std::vector<float> _h;
    std::vector<float> _c;

    std::shared_ptr<Denoiser> _denoiser = nullptr;
    std::shared_ptr<Resampler> _resampler = nullptr;
    std::shared_ptr<SampleQueue> _sample_queue = nullptr;
};

#endif  // VAD_VAD_MODEL_H_