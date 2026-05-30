#include "inference.hpp"
#include <opencv2/opencv.hpp>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <csignal>
#include <thread>
#include <chrono>

using json = nlohmann::json;

static bool running = true;
void handle_sig(int) { running = false; }

// Safe ZMQ send — explicit memcpy, no zero-copy pointer risk
static void zmq_send_tensor(zmq::socket_t& sock,
                             const ActivationTensor& t,
                             int frame_n,
                             double ms) {
    // Part 1: JSON header
    json hdr = {
        {"name",         t.name},
        {"shape",        t.shape},
        {"frame",        frame_n},
        {"inference_ms", ms}
    };
    std::string hdr_str = hdr.dump();
    zmq::message_t hdr_msg(hdr_str.size());
    memcpy(hdr_msg.data(), hdr_str.data(), hdr_str.size());

    // Part 2: raw float32 bytes — explicit copy into ZMQ buffer
    size_t byte_size = t.data.size() * sizeof(float);
    zmq::message_t dat_msg(byte_size);
    memcpy(dat_msg.data(), t.data.data(), byte_size);

    sock.send(hdr_msg, zmq::send_flags::sndmore);
    sock.send(dat_msg, zmq::send_flags::none);
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sig);

    std::string mlpkg = "models/pretrained/mobilenetv2_instrumented.mlpackage";
    if (argc > 1) mlpkg = argv[1];

    std::cout << "[Engine] Loading: " << mlpkg << "\n";
    InferenceEngine engine(mlpkg);
    if (!engine.is_loaded()) {
        std::cerr << "[Engine] Model load FAILED.\n"; return 1;
    }
    std::cout << "[Engine] Model ready.\n";

    // ZMQ PUSH socket
    zmq::context_t ctx(1);
    zmq::socket_t  pub(ctx, ZMQ_PUSH);
    pub.bind("tcp://*:5555");
    std::cout << "[Engine] ZMQ bound on tcp://*:5555\n";

    // Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "[Engine] Webcam failed.\n"; return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    std::cout << "[Engine] Webcam open. Ctrl+C to stop.\n";

    // Warmup — NO cv::waitKey (crashes headless on macOS)
    std::cout << "[Engine] Warming up webcam (5 frames)...\n";
    for (int i = 0; i < 5; i++) {
        cv::Mat tmp;
        cap.read(tmp);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    std::cout << "[Engine] Webcam ready.\n\n";

    int frame_n = 0;
    while (running) {
        cv::Mat bgr;
        if (!cap.read(bgr) || bgr.empty()) continue;

        // BGR → RGB → resize 224×224
        cv::Mat rgb, resized;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(224, 224));

        // Ensure contiguous memory
        if (!resized.isContinuous())
            resized = resized.clone();

        InferenceResult result = engine.predict(resized.data, 224, 224);
        if (!result.success || result.tensors.empty()) continue;

        frame_n++;
        std::cout << "\r[Engine] Frame " << frame_n
                  << "  inf=" << result.inference_ms << "ms"
                  << "  tensors=" << result.tensors.size()
                  << "         " << std::flush;

        // Send each tensor safely
        for (const auto& t : result.tensors) {
            zmq_send_tensor(pub, t, frame_n, result.inference_ms);
        }
    }

    std::cout << "\n[Engine] Stopped.\n";
    cap.release();
    return 0;
}