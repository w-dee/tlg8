#include "image_io.h"

#include <algorithm>
#include <cctype>
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
  std::cerr << "Usage: tlgconv <input.(tlg|tlg5|tlg6|png|bmp)> <output.(tlg|tlg5|tlg6|png|bmp)> [--tlg-version=5|6] [--pixel-format=auto|R8G8B8|A8R8G8B8]\n";
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

  TlgOptions tlgopt; // default version is 6 per requirement
  tlgopt.version = 6;
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
    else if (arg == "-h" || arg == "--help")
    {
      print_usage();
      return 0;
    }
    else
    {
      std::cerr << "Unknown option: " << arg << "\n";
      return 2;
    }
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
  else if (has_ext(in_path, "tlg") || has_ext(in_path, "tlg5") || has_ext(in_path, "tlg6"))
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
    ok = save_png(out_path, img, err);
  }
  else if (has_ext(out_path, "bmp"))
  {
    ok = save_bmp(out_path, img, err);
  }
  else if (has_ext(out_path, "tlg") || has_ext(out_path, "tlg5") || has_ext(out_path, "tlg6"))
  {
    ok = save_tlg(out_path, img, tlgopt, err);
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
