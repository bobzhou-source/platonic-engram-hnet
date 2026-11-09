#pragma once
// Host stand-in for lib/hal/HalStorage.h: plain stdio files, absolute paths.
#include <fcntl.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>

class HalFile {
  FILE* file = nullptr;

 public:
  HalFile() = default;
  HalFile(HalFile&& other) : file(std::exchange(other.file, nullptr)) {}
  HalFile& operator=(HalFile&& other) {
    close();
    file = std::exchange(other.file, nullptr);
    return *this;
  }
  HalFile(const HalFile&) = delete;
  HalFile& operator=(const HalFile&) = delete;
  ~HalFile() { close(); }
  explicit operator bool() const { return file != nullptr; }
  bool openMode(const char* path, const char* mode) {
    close();
    file = fopen(path, mode);
    return file != nullptr;
  }
  bool close() {
    if (file) fclose(file);
    file = nullptr;
    return true;
  }
  int read(void* out, size_t n) { return file ? static_cast<int>(fread(out, 1, n, file)) : -1; }
  size_t write(const void* data, size_t n) { return file ? fwrite(data, 1, n, file) : 0; }
  bool seekSet(size_t pos) { return file && fseek(file, static_cast<long>(pos), SEEK_SET) == 0; }
  size_t fileSize() {
    if (!file) return 0;
    const long pos = ftell(file);
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, pos, SEEK_SET);
    return static_cast<size_t>(size);
  }
};

struct HostStorage {
  bool exists(const char* path) { return path && std::filesystem::exists(path); }
  bool openFileForRead(const char*, const char* path, HalFile& file) { return file.openMode(path, "rb"); }
  bool openFileForRead(const char*, const std::string& path, HalFile& file) {
    return file.openMode(path.c_str(), "rb");
  }
  HalFile open(const char* path, int flags) {
    HalFile file;
    if ((flags & O_TRUNC) || (flags & O_CREAT && !exists(path))) {
      file.openMode(path, "w+b");
    } else {
      file.openMode(path, (flags & (O_RDWR | O_WRONLY)) ? "r+b" : "rb");
    }
    return file;
  }
  bool remove(const char* path) { return std::remove(path) == 0; }
  bool rename(const char* oldPath, const char* newPath) { return std::rename(oldPath, newPath) == 0; }
};
inline HostStorage Storage;
