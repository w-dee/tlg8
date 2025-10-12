#include "image_io.h"
#include "tlg7_io.h"
#include "tlg8_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

static void to_lower_inplace(std::string &s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                 { return (char)std::tolower(c); });
}

bool has_ext(const std::string &path, const char *extLowerNoDot)
{
  auto pos = path.find_last_of('.');
  if (pos == std::string::npos)
    return false;
  std::string e = path.substr(pos + 1);
  to_lower_inplace(e);
  return e == extLowerNoDot;
}

static void print_usage()
{
  std::cerr << "Usage: tlgconv <input.(tlg|tlg5|tlg6|tlg7|tlg8|png|bmp)> <output.(tlg|tlg5|tlg6|tlg7|tlg8|png|bmp)>"
            << " [--tlg-version=5|6|7|8] [--pixel-format=auto|R8G8B8|A8R8G8B8]"
            << " [--tlg7-golomb-table=<path>] [--tlg8-golomb-table=<path>] [-tlg7-dump-residuals=<path>]"
            << " [--tlg7-dump-residuals-order=before|after] [-tlg8-dump-residuals=<path>]"
            << " [--tlg8-dump-residuals-order=predictor|color|hilbert]"
            << " [--tlg8-write-residuals-bmp=<path>]"
            << " [--tlg8-write-residuals-order=predictor|color|hilbert]"
            << " [--tlg8-write-residuals-emphasis=<F>]"
            << " [--tlg8-dump-golomb-prediction=<path>]"
            << " [--print-entropy-bits]"
            << " [--tlg7-order=predictor-first|filter-first]\n";
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    print_usage();
    return 2;
  }
  std::string in_path = argv[1];
  std::string out_path = argv[2];

  TlgOptions tlgopt; // default version is 8 per requirement
  tlgopt.version = 8;
  tlgopt.fmt = ImageFormat::Auto;

  for (int i = 3; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg.rfind("--tlg-version=", 0) == 0)
    {
      std::string v = arg.substr(14);
      if (v == "5")
        tlgopt.version = 5;
      else if (v == "6")
        tlgopt.version = 6;
      else if (v == "7")
        tlgopt.version = 7;
      else if (v == "8")
        tlgopt.version = 8;
      else
      {
        std::cerr << "Invalid --tlg-version: " << v << "\n";
        return 2;
      }
    }
    else if (arg.rfind("--pixel-format=", 0) == 0)
    {
      std::string f = arg.substr(15);
      if (f == "auto")
        tlgopt.fmt = ImageFormat::Auto;
      else if (f == "R8G8B8")
        tlgopt.fmt = ImageFormat::R8G8B8;
      else if (f == "A8R8G8B8")
        tlgopt.fmt = ImageFormat::A8R8G8B8;
      else
      {
        std::cerr << "Invalid --pixel-format: " << f << "\n";
        return 2;
      }
    }
    else if (arg.rfind("--tlg7-order=", 0) == 0)
    {
      std::string order = arg.substr(13);
      to_lower_inplace(order);
      if (order == "predictor-first" || order == "prediction-first" || order == "predictor-then-filter")
        tlgopt.tlg7_pipeline_order = TlgOptions::Tlg7PipelineOrder::PredictorThenFilter;
      else if (order == "filter-first" || order == "filter-then-predictor")
        tlgopt.tlg7_pipeline_order = TlgOptions::Tlg7PipelineOrder::FilterThenPredictor;
      else
      {
        std::cerr << "Invalid --tlg7-order: " << order << "\n";
        return 2;
      }
    }
    else if (arg.rfind("--tlg7-golomb-table=", 0) == 0)
    {
      tlgopt.tlg7_golomb_table_path = arg.substr(20);
    }
    else if (arg.rfind("--tlg8-golomb-table=", 0) == 0)
    {
      tlgopt.tlg8_golomb_table_path = arg.substr(20);
    }
    else if ((arg.rfind("-tlg7-dump-residuals=", 0) == 0) || (arg.rfind("--tlg7-dump-residuals=", 0) == 0))
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid -tlg7-dump-residuals option\n";
        return 2;
      }
      tlgopt.tlg7_dump_residuals_path = arg.substr(eq + 1);
    }
    else if ((arg.rfind("--tlg7-dump-residuals-order=", 0) == 0) ||
             (arg.rfind("-tlg7-dump-residuals-order=", 0) == 0))
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid --tlg7-dump-residuals-order option\n";
        return 2;
      }
      std::string order = arg.substr(eq + 1);
      to_lower_inplace(order);
      if (order == "before")
        tlgopt.tlg7_dump_residuals_order = TlgOptions::DumpResidualsOrder::BeforeHilbert;
      else if (order == "after")
        tlgopt.tlg7_dump_residuals_order = TlgOptions::DumpResidualsOrder::AfterHilbert;
      else
      {
        std::cerr << "Invalid --tlg7-dump-residuals-order: " << order << "\n";
        return 2;
      }
    }
    else if (arg == "-h" || arg == "--help")
    {
      print_usage();
      return 0;
    }
    else if ((arg.rfind("-tlg8-dump-residuals=", 0) == 0) || (arg.rfind("--tlg8-dump-residuals=", 0) == 0))
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid -tlg8-dump-residuals option\n";
        return 2;
      }
      tlgopt.tlg8_dump_residuals_path = arg.substr(eq + 1);
    }
    else if ((arg.rfind("--tlg8-dump-residuals-order=", 0) == 0) ||
             (arg.rfind("-tlg8-dump-residuals-order=", 0) == 0))
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid --tlg8-dump-residuals-order option\n";
        return 2;
      }
      std::string order = arg.substr(eq + 1);
      to_lower_inplace(order);
      if (order == "predictor")
        tlgopt.tlg8_dump_residuals_order = TlgOptions::DumpResidualsOrder::AfterPredictor;
      else if (order == "color")
        tlgopt.tlg8_dump_residuals_order = TlgOptions::DumpResidualsOrder::AfterColorFilter;
      else if (order == "hilbert")
        tlgopt.tlg8_dump_residuals_order = TlgOptions::DumpResidualsOrder::AfterHilbert;
      else
      {
        std::cerr << "Invalid --tlg8-dump-residuals-order: " << order << "\n";
        return 2;
      }
    }
    else if ((arg.rfind("--tlg8-write-residuals-bmp=", 0) == 0) ||
             (arg.rfind("-tlg8-write-residuals-bmp=", 0) == 0))
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid --tlg8-write-residuals-bmp option\n";
        return 2;
      }
      tlgopt.tlg8_write_residuals_bmp_path = arg.substr(eq + 1);
    }
    else if ((arg.rfind("--tlg8-write-residuals-order=", 0) == 0) ||
             (arg.rfind("-tlg8-write-residuals-order=", 0) == 0))
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid --tlg8-write-residuals-order option\n";
        return 2;
      }
      std::string order = arg.substr(eq + 1);
      to_lower_inplace(order);
      if (order == "predictor")
        tlgopt.tlg8_write_residuals_order = TlgOptions::DumpResidualsOrder::AfterPredictor;
      else if (order == "color")
        tlgopt.tlg8_write_residuals_order = TlgOptions::DumpResidualsOrder::AfterColorFilter;
      else if (order == "hilbert")
        tlgopt.tlg8_write_residuals_order = TlgOptions::DumpResidualsOrder::AfterHilbert;
      else
      {
        std::cerr << "Invalid --tlg8-write-residuals-order: " << order << "\n";
        return 2;
      }
    }
    else if (arg.rfind("--tlg8-dump-golomb-prediction=", 0) == 0)
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid --tlg8-dump-golomb-prediction option\n";
        return 2;
      }
      tlgopt.tlg8_dump_golomb_prediction_path = arg.substr(eq + 1);
    }
    else if ((arg.rfind("--tlg8-write-residuals-emphasis=", 0) == 0) ||
             (arg.rfind("-tlg8-write-residuals-emphasis=", 0) == 0))
    {
      const auto eq = arg.find('=');
      if (eq == std::string::npos || eq + 1 >= arg.size())
      {
        std::cerr << "Invalid --tlg8-write-residuals-emphasis option\n";
        return 2;
      }
      const std::string value = arg.substr(eq + 1);
      char *endptr = nullptr;
      const double emphasis = std::strtod(value.c_str(), &endptr);
      if (endptr == value.c_str() || (endptr && *endptr != '\0') || !std::isfinite(emphasis))
      {
        std::cerr << "Invalid --tlg8-write-residuals-emphasis: " << value << "\n";
        return 2;
      }
      tlgopt.tlg8_write_residuals_emphasis = emphasis;
    }
    else if (arg == "--print-entropy-bits")
    {
      tlgopt.print_entropy_bits = true;
    }
    else
    {
      std::cerr << "Unknown option: " << arg << "\n";
      return 2;
    }
  }

  if (tlgopt.print_entropy_bits && tlgopt.version != 8)
  {
    std::cerr << "--print-entropy-bits は TLG8 エンコード時のみ使用できます\n";
    return 2;
  }

  std::string cfg_err;
  if (!tlg::v7::configure_golomb_table(tlgopt.tlg7_golomb_table_path, cfg_err))
  {
    std::cerr << cfg_err << "\n";
    return 1;
  }

  if (!tlg::v8::configure_golomb_table(tlgopt.tlg8_golomb_table_path, cfg_err))
  {
    std::cerr << cfg_err << "\n";
    return 1;
  }

  // Load input image
  PixelBuffer img;
  std::string err;
  bool ok = false;
  if (has_ext(in_path, "png"))
  {
    ok = load_png(in_path, img, err);
  }
  else if (has_ext(in_path, "bmp"))
  {
    ok = load_bmp(in_path, img, err);
  }
  else if (has_ext(in_path, "tlg") || has_ext(in_path, "tlg5") || has_ext(in_path, "tlg6") || has_ext(in_path, "tlg7") || has_ext(in_path, "tlg8"))
  {
    ok = load_tlg(in_path, img, err);
  }
  else
  {
    std::cerr << "Unsupported input extension: " << in_path << "\n";
    return 2;
  }
  if (!ok)
  {
    std::cerr << "Failed to load input: " << err << "\n";
    return 1;
  }

  // Decide pixel format if Auto when writing TLG
  if (tlgopt.fmt == ImageFormat::Auto)
  {
    tlgopt.fmt = img.has_alpha() ? ImageFormat::A8R8G8B8 : ImageFormat::R8G8B8;
  }

  // Save output
  err.clear();
  if (has_ext(out_path, "png"))
  {
    if (tlgopt.print_entropy_bits)
    {
      std::cerr << "--print-entropy-bits は TLG 出力時のみ指定できます\n";
      return 2;
    }
    ok = save_png(out_path, img, err);
  }
  else if (has_ext(out_path, "bmp"))
  {
    if (tlgopt.print_entropy_bits)
    {
      std::cerr << "--print-entropy-bits は TLG 出力時のみ指定できます\n";
      return 2;
    }
    ok = save_bmp(out_path, img, err);
  }
  else if (has_ext(out_path, "tlg") || has_ext(out_path, "tlg5") || has_ext(out_path, "tlg6") || has_ext(out_path, "tlg7") || has_ext(out_path, "tlg8"))
  {
    uint64_t entropy_bits = 0;
    ok = save_tlg(out_path, img, tlgopt, err, tlgopt.print_entropy_bits ? &entropy_bits : nullptr);
    if (ok && tlgopt.print_entropy_bits)
      std::cout << "entropy_bits=" << entropy_bits << "\n";
  }
  else
  {
    std::cerr << "Unsupported output extension: " << out_path << "\n";
    return 2;
  }
  if (!ok)
  {
    std::cerr << "Failed to save output: " << err << "\n";
    return 1;
  }

  return 0;
}
