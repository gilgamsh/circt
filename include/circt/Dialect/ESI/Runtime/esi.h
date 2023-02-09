#pragma once

#include <any>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace circt {
namespace esi {
namespace runtime {

// ESI CPP API goes here. This is general for all backends.

// Bas class for all ports.
template <typename TBackend>
class Port {

public:
  Port(const std::vector<std::string> &clientPath, TBackend &backend,
       const std::string &implType)
      : backend(&backend), clientPath(clientPath) {}

  // Initializes this port - have to do it post-construction due to initBackend
  // being pure virtual.
  void init() { initBackend(); }

  // Hook for the backend port implementation to initialize the port.
  virtual void initBackend() = 0;

protected:
  TBackend *backend = nullptr;
  std::vector<std::string> clientPath;
};

template <typename WriteType, typename ReadType, typename TBackend>
class ReadWritePort
    : public TBackend::template ReadWritePort<WriteType, ReadType> {
public:
  using Impl = typename TBackend::template ReadWritePort<WriteType, ReadType>;
  static_assert(std::is_base_of<runtime::Port<TBackend>, Impl>::value,
                "Backend port must be a subclass of runtime::Port");

  auto operator()(WriteType arg) { return getImpl()->operator()(arg); }
  Impl *getImpl() { return static_cast<Impl *>(this); }

  ReadWritePort(const std::vector<std::string> &clientPath, TBackend &backend,
                const std::string &implType)
      : Impl(clientPath, backend, implType) {}
};

template <typename WriteType, typename TBackend>
class WritePort : public TBackend::template WritePort<WriteType> {
public:
  using Impl = typename TBackend::template WritePort<WriteType>;
  static_assert(std::is_base_of<runtime::Port<TBackend>, Impl>::value,
                "Backend port must be a subclass of runtime::Port");

  auto operator()(WriteType arg) { return getImpl()->operator()(arg); }
  Impl *getImpl() { return static_cast<Impl *>(this); }

  WritePort(const std::vector<std::string> &clientPath, TBackend &backend,
            const std::string &implType)
      : Impl(clientPath, backend, implType) {}
};

template <typename TBackend>
class Module {
public:
  Module(const std::vector<std::shared_ptr<Port<TBackend>>> &ports)
      : ports(ports) {}

  // Initializes this module
  void init() {
    // Initialize all ports
    for (auto &port : ports)
      port->init();
  }

protected:
  std::vector<std::shared_ptr<Port<TBackend>>> ports;
};

} // namespace runtime
} // namespace esi
} // namespace circt
