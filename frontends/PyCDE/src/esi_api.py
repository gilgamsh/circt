#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from io import FileIO
import json
import pathlib
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List

__dir__ = pathlib.Path(__file__).parent


def _camel_to_snake(camel: str):
  if camel.upper() == camel:
    return camel.lower()
  return re.sub(r'(?<!^)(?=[A-Z])', '_', camel).lower()


def _get_ports_for_clients(clients):
  # Assemble lists of clients for each service port.
  ports = {}
  for client in clients:
    port = client['port']['inner']
    if port not in ports:
      ports[port] = []
    ports[port].append(client)
  return ports


class SoftwareApiBuilder:
  """Parent class for all software API builders. Defines an interfaces and tries
  to encourage code sharing and API consistency (between languages)."""

  class Module:
    """Bookkeeping about modules."""

    def __init__(self, name: str):
      self.name = name
      self.instances: Dict[str, SoftwareApiBuilder.Module] = {}
      self.services: List[Dict] = []

  def __init__(self, services_json: str):
    """Read in the system descriptor and set up bookkeeping structures."""
    self.services = json.loads(services_json)
    self.types: Dict[str, Dict] = {}
    self.modules: Dict[str, SoftwareApiBuilder.Module] = {}
    self.env = Environment(loader=FileSystemLoader(str(__dir__)),
                           undefined=StrictUndefined)

    # Get all the modules listed in the service hierarchy. Populate their
    # 'instances' properly.
    for top in self.services["top_levels"]:
      top_mod = self._get_module(top["module"][1:])
      for svc in top["services"]:
        parent: SoftwareApiBuilder.Module = top_mod
        for inner_ref in [
            (inst["outer_sym"], inst["inner"]) for inst in svc["instance_path"]
        ]:
          m = self._get_module(inner_ref[0])
          parent.instances[inner_ref[1]] = m
          parent = m

    # For any modules which have services, add them as appropriate.
    for mod in self.services["modules"]:
      m = self._get_module(mod["symbol"])
      for svc in mod["services"]:
        m.services.append(svc)

  def _get_module(self, mod_sym: str):
    """Get a module adding an entry if it doesn't exist."""
    if mod_sym not in self.modules:
      self.modules[mod_sym] = SoftwareApiBuilder.Module(mod_sym)
    return self.modules[mod_sym]

  def build(self, os: FileIO, tmpl_file: str):
    """Output the API (in a pre-determined order) via callbacks. Encourages some
    level of consistency between language APIs."""

    self.env.globals.update(camel_to_snake=_camel_to_snake,
                            get_ports_for_clients=_get_ports_for_clients,
                            get_type_name=self.get_type_name,
                            type_str_of=self.get_str_type)

    self.update_globals()

    template = self.env.get_template(tmpl_file)
    top_levels = [
        self._get_module(t["module"][1:]) for t in self.services["top_levels"]
    ]
    os.write(
        template.render(services=self.services,
                        modules=self.modules.values(),
                        types=self.types,
                        tops=top_levels))

  def get_type_name(self, type: Dict):
    """Create a name for 'type', record it, and return it."""
    if "capnp_name" in type:
      name = type["capnp_name"]
    else:
      name = "".join([c if c.isalnum() else '_' for c in type["mlir_name"]])
    self.types[name] = type
    return name

  def get_str_type(self, type: Dict):
    assert False, "unimplemented"


def _get_cpp_struct_fields(type_dict):

  @dataclass
  class StructField:
    name: str
    type: str

  struct_fields = []

  def cpp_type(type: Dict, is_base: bool, field_name="data"):
    dialect = type["dialect"]
    mn: str = type["mnemonic"]
    if dialect == "esi" and mn == "channel":
      return cpp_type(type["inner"], is_base)
    if dialect == "builtin":
      if mn.startswith("i") or mn.startswith("ui"):
        width = int(mn.strip("ui"))
        signed = False
      elif mn.startswith("si"):
        width = int(mn.strip("si"))
        signed = True
      if width == 0:
        # Void data in CPP?... choose to not materialize the field.
        return
      else:
        s = "s" if signed else "u"
        if width == 1:
          typestr = "bool"
        elif width <= 8:
          typestr = f"{s}int8_t"
        elif width <= 16:
          typestr = f"{s}int16_t"
        elif width <= 32:
          typestr = f"{s}int32_t"
        elif width <= 64:
          typestr = f"{s}int64_t"
        else:
          assert False, f"unimplemented width {width}"

      if is_base:
        # TODO: should also serialize the field name for unary types in the
        # services.json file. If not, we just have to assume that the Capnp field
        # name is "i".
        field_name = "i"
      struct_fields.append(StructField(name=field_name, type=typestr))
      return
    elif dialect == "hw":
      if mn == "struct":
        assert is_base, "nested structs not supported"
        for subfield in type["fields"]:
          cpp_type(subfield["type"], False, subfield["name"])
        return

    assert False, "unimplemented type"

  cpp_type(type_dict["type_desc"], is_base=True)
  return struct_fields


def _get_cpp_type(type_name, type_dict):

  class CPPType:

    def __init__(self, type_name, type_dict):
      self.name = type_name
      self.fields = _get_cpp_struct_fields(type_dict)

    def is_unary(self):
      # Returns true if this is a unary type (has only a single field).
      return len(self.fields) == 1

    def unary_field(self):
      assert self.is_unary(), "not unary"
      return self.fields[0]

  return CPPType(type_name, type_dict)


class CPPApiBuilder(SoftwareApiBuilder):

  def __init__(self, services_json: str):
    super().__init__(services_json)

  def update_globals(self):
    self.env.globals.update(get_cpp_type=_get_cpp_type)
    self.env.globals.update(get_cpp_port=lambda port: self._get_cpp_port(port))

  def _get_cpp_port(self, port):

    @dataclass
    class CPPPort:
      name: str
      type: str

    # TODO: this is the end-point for now. Seems like we're missing some
    # meta-information to be able to couple the port declarations to the
    # C++ typenames we generated earlier (e.g. the mlir typename of the
    # port types).

    return CPPPort("FIXME", "FIXME")

    to_client_type = self.get_type_name(
        port['to-client-type']) if 'to-client-type' in port else None
    to_server_type = self.get_type_name(
        port['to-server-type']) if 'to-server-type' in port else None

    if to_client_type and to_server_type:
      port_type = f"ReadWritePort</*WriteType:*/ {to_server_type}, /*ReadType:*/ {to_client_type}>"
    elif to_client_type:
      port_type = f"ReadPort</*ReadType:*/ {to_client_type}>"
    elif to_server_type:
      port_type = f"WritePort</*WriteType:*/ {to_server_type}>"
    else:
      assert False, "unimplemented port type"

    return CPPPort(port["name"], port_type)

  def build(self, system_name: str, sw_dir: pathlib.Path):
    """Emit a C++ ESI runtime library into 'output_dir'."""
    libdir = sw_dir / system_name
    if not libdir.exists():
      libdir.mkdir()

    self.scan_types()

    # Emit the system-specific API.
    main = libdir / "api.h"
    super().build(main.open("w"), "esi_api.h.j2")

  def get_str_type(self, type_dict: Dict):
    return "foo"

  def scan_types(self):
    # Iterates through all of the top-level types in the design and records
    # them. CPP needs to declare its types before emitting the modules, so it
    # can't rely on the assumption that get_type_name was called prior to
    # type references.
    for mod in self.modules.values():
      for svc in mod.services:
        ports = _get_ports_for_clients(svc["clients"])
        for _, port_clients in ports.items():
          for pc in port_clients:
            if "to_server_type" in pc:
              self.get_type_name(pc["to_server_type"])
            if "to_client_type" in pc:
              self.get_type_name(pc["to_client_type"])


class PythonApiBuilder(SoftwareApiBuilder):

  def __init__(self, services_json: str):
    super().__init__(services_json)

  def update_globals(self):
    pass

  def build(self, system_name: str, sw_dir: pathlib.Path):
    """Emit a Python ESI runtime library into 'output_dir'."""
    libdir = sw_dir / system_name
    if not libdir.exists():
      libdir.mkdir()

    common_file = libdir / "common.py"
    shutil.copy(__dir__ / "esi_runtime_common.py", common_file)

    # Emit the system-specific API.
    main = libdir / "__init__.py"
    super().build(main.open("w"), "esi_api.py.j2")

  def get_str_type(self, type_dict: Dict):
    """Get a Python code string instantiating 'type'."""

    def py_type(type: Dict):
      dialect = type["dialect"]
      mn: str = type["mnemonic"]
      if dialect == "esi" and mn == "channel":
        return py_type(type["inner"])
      if dialect == "builtin":
        if mn.startswith("i") or mn.startswith("ui"):
          width = int(mn.strip("ui"))
          signed = False
        elif mn.startswith("si"):
          width = int(mn.strip("si"))
          signed = True
        if width == 0:
          return "VoidType()"
        return f"IntType({width}, {signed})"
      elif dialect == "hw":
        if mn == "struct":
          fields = [
              f"('{x['name']}', {py_type(x['type'])})" for x in type["fields"]
          ]
          fields_str = ", ".join(fields)
          return "StructType([" + fields_str + "])"

      assert False, "unimplemented type"

    return py_type(type_dict["type_desc"])
