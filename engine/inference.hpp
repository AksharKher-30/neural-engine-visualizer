#pragma once
#include <string>
#include <vector>

struct ActivationTensor {
    std::string          name;
    std::vector<int64_t> shape;
    std::vector<float>   data;
};

struct InferenceResult {
    std::vector<ActivationTensor> tensors;
    double inference_ms;
    bool   success;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& mlpackage_path);
    ~InferenceEngine();

    // rgb: uint8 H×W×3 pixels, already resized to 224×224
    InferenceResult predict(const uint8_t* rgb, int width, int height);

    bool is_loaded() const { return loaded_; }

private:
    void* objc_engine_;
    bool  loaded_;
};