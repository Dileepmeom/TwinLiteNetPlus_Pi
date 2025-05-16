# TwinLiteNetPlus_CPP

## Build Instructions

## to use latest cmake if needed
export PATH=$HOME/Downloads/cmake-4.0.2-linux-aarch64/bin:$PATH

1. Navigate to the project directory and create a build folder:
   ```bash
   mkdir build && cd build

   cmake .. -DONNXRUNTIME_ROOT=/home/ridebuddy/Documents/libraries/onnxruntime-linux-x64-1.21.0

   make -j$(nproc)

## Inference instructions
./twinlite_infer ../pretrained/small.onnx ../inference/images ../results 

Note :
../pretrained/small.onnx: Path to the ONNX model.
../inference/images: Directory containing input images.
../results: Directory where the output will be saved.