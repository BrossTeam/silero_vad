#include <vector>

#include "gflags/gflags.h"

#include "front_vad/wav.h"
#include "vad/vad_model.h"

#include <iostream>

DEFINE_string(wav_path, "C:/Users/novel/Documents/Sound Recordings/[MONO]-Recording.wav", "wav path");
DEFINE_double(threshold, 0.5, "threshold of voice activity detection");
DEFINE_string(model_path, "data/silero_vad.onnx", "voice activity detection model path");

int dataset_test()
{
    
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    wav::WavReader wav_reader(FLAGS_wav_path);
    int num_channels = wav_reader.num_channels();
    CHECK_EQ(num_channels, 1) << "Only support mono (1 channel) wav!";
    int sample_rate = wav_reader.sample_rate();
    const float* pcm = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    std::vector<float> input_wav{ pcm, pcm + num_samples };

    bool denoise = true;
    float min_sil_dur = 0;  // in seconds
    float speech_pad = 0;   // in seconds
    VadModel vad(FLAGS_model_path, denoise, sample_rate, FLAGS_threshold,
        min_sil_dur, speech_pad);

    std::vector<float> start_pos;
    std::vector<float> stop_pos;
    float dur = vad.Vad(input_wav, &start_pos, &stop_pos);

    if (!stop_pos.empty() && stop_pos.back() > dur) {
        stop_pos.back() = dur;
    }
    if (stop_pos.size() < start_pos.size()) {
        stop_pos.emplace_back(dur);
    }
    for (int i = 0; i < start_pos.size(); i++) {
        std::cout << "[" << start_pos[i] << ", " << stop_pos[i] << "]s"
                  << "\n";
    }
}