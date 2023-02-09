// REQUIRES: esi-cosim

// clang-format off

// Create ESI system
// RUN: rm -rf %t
// RUN: %PYTHON% %S/../esi_ram.py %t 2>&1

// Create ESI CPP API
// ...
// RUN: cp %S/CMakeLists.txt %t
// RUN: cp %s %t
// RUN: cmake -S %t -B %T/build -DCIRCT_DIR=%CIRCT_SOURCE%

// Run test
// RN: esi-cosim-runner.py --tmpdir %t --schema %t/hw/schema.capnp %s %t/hw/*.sv
// RN: ./%T/build/esi_ram_test %t %t/hw/schema.capnp

// clang-format on
#include <any>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <capnp/ez-rpc.h>

#include "hw/schema.capnp.h"

namespace ESICPP {

// Custom type to hold the interface descriptions because i can't for the life
// of me figure out how to cleanly keep capnproto messages around...
struct EsiDpiInterfaceDesc {
  std::string endpointID;
  uint64_t sendTypeID;
  uint64_t recvTypeID;
};

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
  static_assert(std::is_base_of<ESICPP::Port<TBackend>, Impl>::value,
                "Backend port must be a subclass of ESICPP::Port");

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
  static_assert(std::is_base_of<ESICPP::Port<TBackend>, Impl>::value,
                "Backend port must be a subclass of ESICPP::Port");

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

} // namespace ESICPP

namespace esi_cosim {
// ESI cosim backend goes here.

template <typename WriteType, typename ReadType>
class CosimReadWritePort;

template <typename WriteType>
class CosimWritePort;

class CosimBackend {
public:
  // Using directives to point the base class implementations to the cosim
  // port implementations.

  template <typename WriteType, typename ReadType>
  using ReadWritePort = CosimReadWritePort<WriteType, ReadType>;

  template <typename WriteType>
  using WritePort = CosimWritePort<WriteType>;

  CosimBackend(const std::string &host, uint64_t hostPort) {
    ezClient = std::make_unique<capnp::EzRpcClient>(host, hostPort);
    dpiClient = std::make_unique<CosimDpiServer::Client>(
        ezClient->getMain<CosimDpiServer>());

    list();
  }

  // Returns a list of all available endpoints.
  const std::vector<ESICPP::EsiDpiInterfaceDesc> &list() {
    if (endpoints.has_value())
      return *endpoints;

    // Query the DPI server for a list of available endpoints.
    auto listReq = dpiClient->listRequest();
    auto ifaces = listReq.send().wait(ezClient->getWaitScope()).getIfaces();
    endpoints = std::vector<ESICPP::EsiDpiInterfaceDesc>();
    for (auto iface : ifaces) {
      ESICPP::EsiDpiInterfaceDesc desc;
      desc.endpointID = iface.getEndpointID().cStr();
      desc.sendTypeID = iface.getSendTypeID();
      desc.recvTypeID = iface.getRecvTypeID();
      endpoints->push_back(desc);
    }

    // print out the endpoints
    for (auto ep : *endpoints) {
      std::cout << "Endpoint: " << ep.endpointID << std::endl;
      std::cout << "  Send Type: " << ep.sendTypeID << std::endl;
      std::cout << "  Recv Type: " << ep.recvTypeID << std::endl;
    }

    return *endpoints;
  }

  template <typename CnPWriteType, typename CnPReadType>
  auto getPort(const std::vector<std::string> &clientPath) {
    // Join client path into a single string with '.' as a separator.
    std::string clientPathStr;
    for (auto &path : clientPath) {
      if (!clientPathStr.empty())
        clientPathStr += '.';
      clientPathStr += path;
    }

    // Everything is nested under "TOP.top"
    clientPathStr = "TOP.top." + clientPathStr;

    auto openReq = dpiClient->openRequest<CnPWriteType, CnPReadType>();

    // Scan through the available endpoints to find the requested one.
    bool found = false;
    for (auto &ep : list()) {
      auto epid = ep.endpointID;
      if (epid == clientPathStr) {
        auto iface = openReq.getIface();
        iface.setEndpointID(epid);
        iface.setSendTypeID(ep.sendTypeID);
        iface.setRecvTypeID(ep.recvTypeID);
        found = true;
        break;
      }
    }

    if (!found) {
      throw std::runtime_error("Could not find endpoint: " + clientPathStr);
    }

    // Open the endpoint.
    auto openResp = openReq.send().wait(ezClient->getWaitScope());
    return openResp.getIface();
  }

  bool supportsImpl(const std::string &implType) {
    // The cosim backend only supports cosim connectivity implementations
    return implType == "cosim";
  }

  kj::WaitScope &getWaitScope() { return ezClient->getWaitScope(); }

protected:
  std::unique_ptr<capnp::EzRpcClient> ezClient;
  std::unique_ptr<CosimDpiServer::Client> dpiClient;
  std::optional<std::vector<ESICPP::EsiDpiInterfaceDesc>> endpoints;
};

template <typename WriteType, typename ReadType>
class CosimReadWritePort : public ESICPP::Port<CosimBackend> {
  using BasePort = ESICPP::Port<CosimBackend>;

public:
  CosimReadWritePort(const std::vector<std::string> &clientPath,
                     CosimBackend &backend, const std::string &implType)
      : BasePort(clientPath, backend, implType) {}

  ReadType operator()(WriteType arg) {
    auto req = port->sendRequest();
    arg.fillCapnp(req.getMsg());
    req.send().wait(this->backend->getWaitScope());
    std::optional<capnp::Response<typename EsiDpiEndpoint<
        typename WriteType::CPType, typename ReadType::CPType>::RecvResults>>
        resp;
    do {
      auto recvReq = port->recvRequest();
      recvReq.setBlock(false);

      resp = recvReq.send().wait(this->backend->getWaitScope());
    } while (!resp->getHasData());
    auto data = resp->getResp();
    return ReadType::fromCapnp(data);
  }

  void initBackend() override {
    port =
        backend->getPort<typename WriteType::CPType, typename ReadType::CPType>(
            clientPath);
  }

private:
  // Handle to the underlying endpoint.
  std::optional<typename ::EsiDpiEndpoint<typename WriteType::CPType,
                                          typename ReadType::CPType>::Client>
      port;
};

template <typename WriteType>
class CosimWritePort : public ESICPP::Port<CosimBackend> {
  using BasePort = ESICPP::Port<CosimBackend>;

public:
  CosimWritePort(const std::vector<std::string> &clientPath,
                 CosimBackend &backend, const std::string &implType)
      : BasePort(clientPath, backend, implType) {}

  void initBackend() override {
    port = backend->getPort<typename WriteType::CPType, ::I1>(clientPath);
  }

  void operator()(WriteType arg) {
    auto req = port->sendRequest();
    arg.fillCapnp(req.getMsg());
    req.send().wait(this->backend->getWaitScope());
  }

private:
  // Handle to the underlying endpoint.
  std::optional<
      typename ::EsiDpiEndpoint<typename WriteType::CPType, ::I1>::Client>
      port;
};

} // namespace esi_cosim

namespace ESIMem {

// Generated things for the current ESI system.

// "pretty" types to wrap Cap'n Proto madness types. Can easily be autogenerated
// based on the input schema.
struct I1 {
  // Data members.
  bool i;

  I1(bool i) : i(i) {}
  I1() = default;

  // Unary types have convenience conversion operators.
  operator bool() const { return i; }

  // Spaceship operator for comparison convenience.
  auto operator<=>(const I1 &) const = default;

  // Generated sibling type.
  using CPType = ::I1;
  void fillCapnp(CPType::Builder cp) { cp.setI(i); }
  static I1 fromCapnp(CPType::Reader msg) { return I1(msg.getI()); }
};

struct I3 {
  using CPType = ::I3;
  uint8_t i;

  // Convenience constructor due to unary type (allows implicit conversion from
  // literals, makes the API a bit less verbose).
  I3(uint8_t i) : i(i) {}
  I3() = default;

  operator uint8_t() const { return i; }
  auto operator<=>(const I3 &) const = default;

  void fillCapnp(CPType::Builder cp) { cp.setI(i); }
  static I3 fromCapnp(CPType::Reader msg) { return I3(msg.getI()); }
};

struct I64 {
  using CPType = ::I64;
  uint64_t i;
  // use default constructors for all types.
  I64(uint64_t i) : i(i) {}
  I64(int i) : i(i) {}
  I64() = default;
  auto operator<=>(const I64 &) const = default;

  operator uint64_t() const { return i; }
  void fillCapnp(CPType::Builder cp) { cp.setI(i); }
  static I64 fromCapnp(CPType::Reader msg) { return I64(msg.getI()); }
};

struct Struct16871797234873963366 {
  using CPType = ::Struct16871797234873963366;
  uint8_t address;
  uint64_t data;

  auto operator<=>(const Struct16871797234873963366 &) const = default;

  void fillCapnp(CPType::Builder cp) {
    cp.setAddress(address);
    cp.setData(data);
  }

  static Struct16871797234873963366 fromCapnp(CPType::Reader msg) {
    return Struct16871797234873963366{.address = msg.getAddress(),
                                      .data = msg.getData()};
  }
};

template <typename TBackend>
class MemComms : public ESICPP::Module<TBackend> {
  using Port = ESICPP::Port<TBackend>;

public:
  // Port type declarations
  using Tread0 = ESICPP::ReadWritePort</*readType=*/ESIMem::I3,
                                       /*writeType=*/ESIMem::I64, TBackend>;
  using Tread0Ptr = std::shared_ptr<Tread0>;

  using Tloopback0 = ESICPP::ReadWritePort<
      /*readType=*/ESIMem::Struct16871797234873963366,
      /*writeType=*/ESIMem::Struct16871797234873963366, TBackend>;
  using Tloopback0Ptr = std::shared_ptr<Tloopback0>;

  using Twrite0 = ESICPP::WritePort<
      /*writeType=*/ESIMem::Struct16871797234873963366, TBackend>;
  using Twrite0Ptr = std::shared_ptr<Twrite0>;

  MemComms(Tread0Ptr read0, Tloopback0Ptr loopback0, Twrite0Ptr write0)
      : ESICPP::Module<TBackend>({read0, loopback0, write0}), read0(read0),
        loopback0(loopback0), write0(write0) {}

  std::shared_ptr<Tread0> read0;
  std::shared_ptr<Tloopback0> loopback0;
  std::shared_ptr<Twrite0> write0;
};

template <typename TBackend>
class Top {

public:
  Top(TBackend &backend) {

    {

      // memComms initialization
      auto read0 = std::make_shared<
          ESICPP::ReadWritePort</*writeType=*/ESIMem::I3,
                                /*readType=*/ESIMem::I64, TBackend>>(
          std::vector<std::string>{"read"}, backend, "cosim");

      auto loopback0 = std::make_shared<ESICPP::ReadWritePort<
          /*writeType=*/ESIMem::Struct16871797234873963366,
          /*readType=*/ESIMem::Struct16871797234873963366, TBackend>>(
          std::vector<std::string>{"loopback"}, backend, "cosim");

      auto write0 = std::make_shared<ESICPP::WritePort<
          /*readType=*/ESIMem::Struct16871797234873963366, TBackend>>(
          std::vector<std::string>{"write"}, backend, "cosim");
      memComms = std::make_unique<MemComms<TBackend>>(read0, loopback0, write0);
      memComms->init();
    };

  }; // namespace ESIMem

  // std::unique_ptr<DeclareRandomAccessMemory<TBackend>> declram;
  std::unique_ptr<MemComms<TBackend>> memComms;
};

} // namespace ESIMem

namespace esi_test {
// Test namespace - this is all user-written code

template <typename TBackend>
int runTest(TBackend &backend) {
  // Connect the ESI system to the provided backend.
  ESIMem::Top top(backend);

  auto write_cmd = ESIMem::Struct16871797234873963366{.address = 2, .data = 42};

  auto loopback_result = (*top.memComms->loopback0)(write_cmd);
  if (loopback_result != write_cmd)
    return 1;

  auto read_result = (*top.memComms->read0)(2);
  if (read_result != ESIMem::I64(0))
    return 2;
  read_result = (*top.memComms->read0)(3);
  if (read_result != ESIMem::I64(0))
    return 3;

  (*top.memComms->write0)(write_cmd);
  read_result = (*top.memComms->read0)(2);
  if (read_result != ESIMem::I64(42))
    return 4;
  read_result = (*top.memComms->read0)(3);
  if (read_result != ESIMem::I64(42))
    return 5;

  // Re-write a 0 to the memory (mostly for debugging purposes to allow us to
  // keep the server alive and rerun the test).
  write_cmd = ESIMem::Struct16871797234873963366{.address = 2, .data = 0};
  (*top.memComms->write0)(write_cmd);
  read_result = (*top.memComms->read0)(2);
  if (read_result != ESIMem::I64(0))
    return 6;

  return 0;
}

int run_cosim_test(const std::string &host, unsigned port) {
  // Run test with cosimulation backend.
  esi_cosim::CosimBackend cosim(host, port);
  return runTest(cosim);
}

} // namespace esi_test

int main(int argc, char **argv) {
  std::string rpchostport;
  if (argc != 2) {
    // Schema not currently used but required by the ESI cosim tester
    std::cerr << "usage: esi_ram_test configfile" << std::endl;
    return 1;
  }

  auto configFile = argv[1];

  // Parse the config file. It contains a line "port : ${port}"
  std::ifstream config(configFile);
  std::string line;
  while (std::getline(config, line)) {
    auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    auto key = line.substr(0, colon);
    auto value = line.substr(colon + 1);
    if (key == "port") {
      rpchostport = "localhost:" + value;
      break;
    }
  }

  if (rpchostport.empty()) {
    std::cerr << "Could not find port in config file" << std::endl;
    return 1;
  }

  auto colon = rpchostport.find(':');
  auto host = rpchostport.substr(0, colon);
  auto port = stoi(rpchostport.substr(colon + 1));

  auto res = esi_test::run_cosim_test(host, port);
  if (res != 0) {
    std::cerr << "Test failed with error code " << res << std::endl;
    return 1;
  }
  std::cout << "Test passed" << std::endl;
  return 0;
}
