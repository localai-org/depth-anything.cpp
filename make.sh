cmake -B build -DDA_BUILD_CLI=ON -DCMAKE_CUDA_ARCHITECTURES=native -DDA_GGML_CUDA=ON
cmake --build build -j
