// twinlite_infer.cpp
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <input_dir> <output_dir>\n";
        return 1;
    }

    const char* model_path = argv[1];
    fs::path input_dir  = argv[2];
    fs::path output_dir = argv[3];

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: input_dir does not exist or is not a directory\n";
        return 1;
    }
    if (fs::exists(output_dir))
        fs::remove_all(output_dir);
    fs::create_directory(output_dir);

    // Initialize ONNX Runtime (CPU)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TwinLite");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    //opts.SetIntraOpNumThreads(1);
    
    // ** Enable XNNPACK EP ** //
    //Ort::SessionOptions::XnnpackProviderOptions xnn_opts;
    //xnn_opts.numThreads = 3;
    //opts.AppendExecutionProvider_Xnnpack(xnn_opts);
    
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path, opts);
    Ort::AllocatorWithDefaultOptions allocator;

    // Fetch input name
    size_t num_inputs = session.GetInputCount();
    std::vector<const char*> input_names(num_inputs);
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    input_name_ptrs.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        input_name_ptrs.push_back(session.GetInputNameAllocated(i, allocator));
        input_names[i] = input_name_ptrs.back().get();
    }

    // Fetch output names (drivable area & lane line)
    size_t num_outputs = session.GetOutputCount();
    std::vector<const char*> output_names(num_outputs);
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    output_name_ptrs.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
        output_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names[i] = output_name_ptrs.back().get();
    }

    // Prepare CPU tensor info
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Constants from Python script
    const int TGT_W = 640, TGT_H = 384;

    // Process each image
    for (auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) {
            std::cerr << "Failed to open " << entry.path() << "\n";
            continue;
        }

        // Resize exactly as Python: no letterbox
        cv::Mat img_rs;
        cv::resize(img, img_rs, cv::Size(TGT_W, TGT_H));

        // Preprocess: BGR->RGB, HWC->CHW, F32/255
        cv::Mat rgb;
        cv::cvtColor(img_rs, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

        // HWC to CHW contiguous buffer
        std::vector<float> input_tensor(TGT_W * TGT_H * 3);
        size_t idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < TGT_H; y++) {
                for (int x = 0; x < TGT_W; x++) {
                    input_tensor[idx++] = rgb.at<cv::Vec3f>(y,x)[c];
                }
            }
        }

        // Create ONNX tensor
        std::array<int64_t,4> shape = {1, 3, TGT_H, TGT_W};
        Ort::Value ort_input = Ort::Value::CreateTensor<float>(
            mem_info,
            input_tensor.data(),
            input_tensor.size(),
            shape.data(),
            shape.size());

        // Inference + timing
        auto t0 = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &ort_input, 1,
            output_names.data(), num_outputs);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Extract raw outputs
        // output_tensors[0] => drivable area logits
        // output_tensors[1] => lane line logits
        // assume both are shape {1,2,H,W}
        auto &da_tensor = output_tensors[0];
        auto &ll_tensor = output_tensors[1];
        auto da_info = da_tensor.GetTensorTypeAndShapeInfo();
        auto ll_info = ll_tensor.GetTensorTypeAndShapeInfo();
        auto da_dims = da_info.GetShape();
        auto ll_dims = ll_info.GetShape();
        int C_da = da_dims[1], H = da_dims[2], W = da_dims[3];
        int C_ll = ll_dims[1];  // should equal 2, H, W same

        const float* da_data = da_tensor.GetTensorData<float>();
        const float* ll_data = ll_tensor.GetTensorData<float>();

        // Build binary masks via argmax
        cv::Mat da_mask(H, W, CV_8UC1), ll_mask(H, W, CV_8UC1);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                // drivable area
                float best = da_data[0*H*W + y*W + x];
                int     cls  = 0;
                for (int c = 1; c < C_da; c++) {
                    float v = da_data[c*H*W + y*W + x];
                    if (v > best) { best = v; cls = c; }
                }
                da_mask.at<uint8_t>(y,x) = (cls == 1 ? 255 : 0);

                // lane line
                best = ll_data[0*H*W + y*W + x];
                int cls2 = 0;
                for (int c = 1; c < C_ll; c++) {
                    float v = ll_data[c*H*W + y*W + x];
                    if (v > best) { best = v; cls2 = c; }
                }
                ll_mask.at<uint8_t>(y,x) = (cls2 == 1 ? 255 : 0);
            }
        }

        // Overlay on img_rs: blue for DA, green for LL
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if (da_mask.at<uint8_t>(y,x) > 100) {
                    img_rs.at<cv::Vec3b>(y,x) = {255, 0, 0};
                } else if (ll_mask.at<uint8_t>(y,x) > 100) {
                    img_rs.at<cv::Vec3b>(y,x) = {0, 255, 0};
                }
            }
        }

        // Save & log
        fs::path outp = output_dir / entry.path().filename();
        cv::imwrite(outp.string(), img_rs);
        std::cout << "Image " << entry.path().filename().string()
                  << ": Inference time = " << std::fixed << std::setprecision(4)
                  << elapsed << " seconds\n";
    }

    return 0;
}
