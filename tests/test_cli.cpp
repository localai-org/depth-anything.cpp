#include "cli.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* message) {
    std::fprintf(stderr, "%s: %s\n", ok ? "ok" : "FAIL", message);
    if (!ok) ++failures;
}

static da::cli::Parsed parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) argv.push_back(arg.data());
    return da::cli::parse(static_cast<int>(argv.size()), argv.data());
}

int main() {
    auto defaults = parse({"da3-cli", "depth-upscale", "--input", "rgb.tiff",
                           "--sensor-depth", "sensor.tiff", "--tiff", "depth.tiff"});
    check(defaults.error.empty(), "depth-upscale accepts required arguments without --model");
    check(defaults.model == "models/depth-anything2-base-f32.gguf",
          "depth-upscale uses the converted DA2 ONNX model by default");

    auto explicit_model = parse({"da3-cli", "depth-upscale", "--model", "custom.gguf",
                                 "--input", "rgb.tiff", "--sensor-depth", "sensor.tiff",
                                 "--tiff", "depth.tiff"});
    check(explicit_model.error.empty(), "depth-upscale accepts an explicit model");
    check(explicit_model.model == "custom.gguf", "explicit depth-upscale model overrides default");
    return failures == 0 ? 0 : 1;
}
