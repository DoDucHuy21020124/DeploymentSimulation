#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
  // Edit these two paths directly when you want to change input/output.
  const std::string input_image_path = "./result.jpg";
  const std::string output_folder = "./output";

  cv::Mat image = cv::imread(input_image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Error: Could not load image: " << input_image_path << "\n";
    return 1;
  }

  std::filesystem::path out_dir(output_folder);
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "Error: Could not create output folder: " << output_folder
              << "\nReason: " << ec.message() << "\n";
    return 1;
  }

  std::filesystem::path in_path(input_image_path);
  std::filesystem::path out_path = out_dir / in_path.filename();

  if (!cv::imwrite(out_path.string(), image)) {
    std::cerr << "Error: Could not save image to: " << out_path << "\n";
    return 1;
  }

  std::cout << "Saved image to: " << out_path << "\n";
  return 0;
}
