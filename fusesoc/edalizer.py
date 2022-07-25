# Copyright FuseSoC contributors
# Licensed under the 2-Clause BSD License, see LICENSE for details.
# SPDX-License-Identifier: BSD-2-Clause

import argparse
import hashlib
import logging
import os
import pathlib
import shutil
from filecmp import cmp

from fusesoc import utils
from fusesoc.coremanager import DependencyError
from fusesoc.librarymanager import Library
from fusesoc.utils import depgraph_to_dot, merge_dict
from fusesoc.vlnv import Vlnv

logger = logging.getLogger(__name__)


class FileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = os.path.expandvars(values[0])
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        setattr(namespace, self.dest, [path])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Edalizer:
    def __init__(
        self,
        toplevel,
        flags,
        work_root,
        core_manager,
        export_root=None,
        system_name=None,
        resolve_env_vars=False,
    ):
        logger.debug("Building EDAM structure")

        self.toplevel = toplevel
        self.flags = flags
        self.core_manager = core_manager
        self.work_root = work_root
        self.export_root = export_root
        self.system_name = system_name
        self.resolve_env_vars = resolve_env_vars

        self.generators = {}
        self._cached_core_list_for_generator = []

    @property
    def cores(self):
        return self.resolved_cores

    @property
    def resolved_cores(self):
        """Get a list of all "used" cores after the dependency resolution"""
        try:
            return self.core_manager.get_depends(self.toplevel, self.flags)
        except DependencyError as e:
            logger.error(
                e.msg + f"\nFailed to resolve dependencies for {self.toplevel}"
            )
            exit(1)
        except SyntaxError as e:
            logger.error(e.msg)
            exit(1)

    @property
    def discovered_cores(self):
        """Get a list of all cores found by fusesoc"""
        return self.core_manager.db.find()

    def run(self):
        """Run all steps to create a EDAM file"""

        # Run the setup task on all cores (fetch and patch them as needed)
        self.setup_cores()

        # Get all generators defined in any of the cores
        self.extract_generators()

        # Run all generators. Generators can create new cores, which are added
        # to the list of available cores.
        self.run_generators()

        # If a work root exists for us to write it to, dump complete dependency
        # tree as dot file
        dot_ext = ".deps-after-generators.dot"
        dot_basename = self.toplevel.sanitized_name + dot_ext
        dot_filepath = os.path.join(self.work_root, dot_basename)
        if not os.path.exists(self.work_root):
            logger.info(
                f"Skipped writing dependency graph to {dot_filepath} "
                "(work root doesn't exist)"
            )
        else:
            core_graph = self.core_manager.get_dependency_graph(
                self.toplevel, self.flags
            )
            with open(dot_filepath, "w") as f:
                f.write(depgraph_to_dot(core_graph))
            logger.info(f"Wrote dependency graph to {dot_filepath}")

        # Create EDA API file contents
        self.create_edam()

        return self.edam

    def _core_flags(self, core):
        """Get flags for a specific core"""

        core_flags = self.flags.copy()
        core_flags["is_toplevel"] = core.name == self.toplevel
        return core_flags

    def setup_cores(self):
        """Setup cores: fetch resources, patch them, etc."""
        for core in self.cores:
            logger.info("Preparing " + str(core.name))
            core.setup()

    def extract_generators(self):
        """Get all registered generators from the cores"""
        generators = {}
        for core in self.cores:
            _flags = self._core_flags(core)
            logger.debug("Searching for generators in " + str(core.name))
            core_generators = core.get_generators(_flags)
            logger.debug(f"Found generators: {core_generators.keys()}")
            generators.update(core_generators)

        self.generators = generators

    def _invalidate_cached_core_list_for_generator(self):
        if self._cached_core_list_for_generator:
            self._cached_core_list_for_generator = None

    def _core_list_for_generator(self):
        """Produce a dictionary of cores, suitable for passing to a generator

        The results of this functions are cached for a significant overall
        speedup. Users need to call _invalidate_cached_core_list_for_generator()
        whenever the CoreDB is modified.
        """

        if self._cached_core_list_for_generator:
            return self._cached_core_list_for_generator

        out = {}
        resolved_cores = self.resolved_cores  # cache for speed
        for core in self.discovered_cores:
            core_flags = self._core_flags(core)
            out[str(core)] = {
                "capi_version": core.capi_version,
                "core_filepath": os.path.abspath(core.core_file),
                "used": core in resolved_cores,
                "core_root": os.path.abspath(core.core_root),
                "files": [str(f["name"]) for f in core.get_files(core_flags)],
            }

        self._cached_core_list_for_generator = out
        return out

    def run_generators(self):
        """Run all generators"""
        generated_libraries = []
        generated_cores = []
        self._generated_core_dirs_to_remove = []
        for core in self.cores:
            logger.debug("Running generators in " + str(core.name))
            core_flags = self._core_flags(core)
            for ttptttg_data in core.get_ttptttg(core_flags):
                ttptttg = Ttptttg(
                    ttptttg_data,
                    core,
                    self.generators,
                    self.work_root if not self.export_root else None,
                    resolve_env_vars=self.resolve_env_vars,
                    core_list=self._core_list_for_generator(),
                    toplevel=str(self.toplevel),
                )

                gen_lib = ttptttg.generate()
                gen_cores = self.core_manager.find_cores(gen_lib, ignored_dirs=[])

                for gen_core in gen_cores:
                    if self.export_root and not (
                        ttptttg.is_generator_cacheable()
                        or ttptttg.is_input_cacheable()
                    ):
                        self._generated_core_dirs_to_remove.append(gen_core.core_root)

                # The output directory of the generator can contain core
                # files, which need to be added to the dependency tree.
                # This isn't done instantly, but only after all generators
                # have finished, to re-do the dependency resolution only
                # once, and not once per generator run.
                generated_libraries.append(gen_lib)

                # Create a dependency to all generated cores.
                # XXX: We need a cleaner API to the CoreManager to add
                # these dependencies. Until then, explicitly use a private
                # API to be reminded that this is a workaround.
                gen_core_vlnvs = [core.name for core in gen_cores]
                logger.debug(
                    "The generator produced the following cores, which "
                    "are inserted into the dependency tree: %s",
                    gen_cores,
                )
                core._generator_created_dependencies += gen_core_vlnvs

                # Collect VLNVs of all generated cores. This information is
                # required to set is_generated for all these cores after
                # adding them to fusesoc. This is needed to later on
                # relocate generator outputs and to delete the ttptttg
                # temporary working directories afterwards.
                generated_cores.extend([str(vlnv) for vlnv in gen_core_vlnvs])

        # Make all new libraries known to fusesoc. This invalidates the solver
        # cache and is therefore quite expensive.
        for lib in generated_libraries:
            self.core_manager.add_library(lib, ignored_dirs=[])
        self._invalidate_cached_core_list_for_generator()

        # Set is_generated for all generated cores.
        cores = self.core_manager.get_cores()
        for core in cores:
            if core in generated_cores:
                logger.debug(f"Setting 'is_generated' for generated core {core}")
                cores[core].is_generated = True

    def export(self):
        for core in self.cores:
            _flags = self._core_flags(core)

            # Export core files
            if self.export_root:
                files_root = os.path.join(self.export_root, core.name.sanitized_name)
                core.export(files_root, _flags)
            else:
                files_root = core.files_root

            # Add copyto files
            for file in core.get_files(_flags):
                if file.get("copyto"):
                    src = os.path.join(files_root, file["name"])
                    self._copyto(src, file.get("copyto"))

        # Clean up ttptttg temporary directories
        self.clean_temp_dirs()

    def _copyto(self, src, name):
        dst = os.path.join(self.work_root, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not os.path.exists(dst) or not cmp(src, dst):
            try:
                shutil.copy2(src, dst)
            except IsADirectoryError:
                shutil.copytree(
                    src,
                    dst,
                    dirs_exist_ok=True,
                )

    def create_edam(self):
        first_snippets = []
        snippets = []
        last_snippets = []
        parameters = {}
        for core in self.cores:
            snippet = {}

            logger.debug("Collecting EDAM parameters from {}".format(str(core.name)))
            _flags = self._core_flags(core)

            # Extract direct dependencies
            snippet["dependencies"] = {str(core.name): core.direct_deps}

            # Extract files
            if self.export_root:
                files_root = os.path.join(self.export_root, core.name.sanitized_name)
            elif core.is_generated:
                files_root = os.path.join(self.work_root, "generated", core.name.sanitized_name)
            else:
                files_root = core.files_root

            rel_root = os.path.relpath(files_root, self.work_root)

            # Extract parameters
            snippet["parameters"] = core.get_parameters(_flags, parameters)
            merge_dict(parameters, snippet["parameters"])

            # Extract tool options
            if self.flags.get("tool"):
                snippet["tool_options"] = {
                    self.flags["tool"]: core.get_tool_options(_flags)
                }

            # Extract flow options
            snippet["flow_options"] = core.get_flow_options(_flags)

            # Extract scripts
            snippet["hooks"] = core.get_scripts(rel_root, _flags)

            _files = []
            for file in core.get_files(_flags):

                # Reparent file path
                file["name"] = str(
                    file.get("copyto", os.path.join(rel_root, file["name"]))
                )

                # Set owning core
                file["core"] = str(core.name)

                # copyto tag shouldn't be in EDAM
                file.pop("copyto", None)

                # Reparent include paths
                if file.get("include_path"):
                    file["include_path"] = os.path.join(rel_root, file["include_path"])

                _files.append(file)

            snippet["files"] = _files

            # Extract VPI modules
            snippet["vpi"] = []
            for _vpi in core.get_vpi(_flags):
                snippet["vpi"].append(
                    {
                        "name": _vpi["name"],
                        "src_files": [
                            os.path.join(rel_root, f) for f in _vpi["src_files"]
                        ],
                        "include_dirs": [
                            os.path.join(rel_root, i) for i in _vpi["include_dirs"]
                        ],
                        "libs": _vpi["libs"],
                    }
                )

            if hasattr(core, "pos"):
                if core.pos == "first":
                    first_snippets.append(snippet)
                elif core.pos == "last":
                    last_snippets.append(snippet)
                elif core.pos == "prepend" and len(snippets) > 0:
                    snippets.insert(len(snippets) - 1, snippet)
                else:
                    snippets.append(snippet)
            else:
                snippets.append(snippet)

        top_core = self.resolved_cores[-1]
        self.edam = {
            "version": "0.2.1",
            "name": self.system_name or top_core.name.sanitized_name,
            "toplevel": top_core.get_toplevel(self.flags),
        }

        for snippet in first_snippets + snippets + last_snippets:
            merge_dict(self.edam, snippet)

    def clean_temp_dirs(self):
        for coredir in self._generated_core_dirs_to_remove:
            logger.debug(f"Removing {coredir} ttptttg temporary directory")
            shutil.rmtree(coredir)

    def _build_parser(self, backend_class, edam):
        typedict = {
            "bool": {"type": str2bool, "nargs": "?", "const": True},
            "file": {"type": str, "nargs": 1, "action": FileAction},
            "int": {"type": int, "nargs": 1},
            "str": {"type": str, "nargs": 1},
            "real": {"type": float, "nargs": 1},
        }
        progname = "fusesoc run {}".format(edam["name"])

        parser = argparse.ArgumentParser(prog=progname, conflict_handler="resolve")
        param_groups = {}
        _descr = {
            "plusarg": "Verilog plusargs (Run-time option)",
            "vlogparam": "Verilog parameters (Compile-time option)",
            "vlogdefine": "Verilog defines (Compile-time global symbol)",
            "generic": "VHDL generic (Run-time option)",
            "cmdlinearg": "Command-line arguments (Run-time option)",
        }
        param_type_map = {}

        paramtypes = backend_class.argtypes
        for name, param in edam["parameters"].items():
            _description = param.get("description", "No description")
            _paramtype = param["paramtype"]
            if _paramtype in paramtypes:
                if not _paramtype in param_groups:
                    param_groups[_paramtype] = parser.add_argument_group(
                        _descr[_paramtype]
                    )

                default = None
                if not param.get("default") is None:
                    try:
                        if param["datatype"] == "bool":
                            default = param["default"]
                        else:
                            default = [
                                typedict[param["datatype"]]["type"](param["default"])
                            ]
                    except KeyError as e:
                        pass
                try:
                    param_groups[_paramtype].add_argument(
                        "--" + name,
                        help=_description,
                        default=default,
                        **typedict[param["datatype"]],
                    )
                except KeyError as e:
                    raise RuntimeError(
                        "Invalid data type {} for parameter '{}'".format(str(e), name)
                    )
                param_type_map[name.replace("-", "_")] = _paramtype
            else:
                logging.warn(
                    "Parameter '{}' has unsupported type '{}' for requested backend".format(
                        name, _paramtype
                    )
                )

        # backend_args.
        backend_args = parser.add_argument_group("Backend arguments")

        if hasattr(backend_class, "get_flow_options"):
            for k, v in backend_class.get_flow_options().items():
                backend_args.add_argument(
                    "--" + k,
                    help=v["desc"],
                    **typedict[v["type"]],
                )
            for k, v in backend_class.get_tool_options(
                self.activated_flow_options
            ).items():
                backend_args.add_argument(
                    "--" + k,
                    help=v["desc"],
                    **typedict[v["type"]],
                )
        else:
            _opts = backend_class.get_doc(0)
            for _opt in _opts.get("members", []) + _opts.get("lists", []):
                backend_args.add_argument("--" + _opt["name"], help=_opt["desc"])

        return parser

    def add_parsed_args(self, backend_class, parsed_args):
        if hasattr(backend_class, "get_flow_options"):
            backend_members = []
            backend_lists = []
            for k, v in backend_class.get_flow_options().items():
                if v.get("list"):
                    backend_lists.append(k)
                else:
                    backend_members.append(k)
            for k, v in backend_class.get_tool_options(
                self.activated_flow_options
            ).items():
                if v.get("list"):
                    backend_lists.append(k)
                else:
                    backend_members.append(k)
            tool_options = self.edam["flow_options"]
        else:
            _opts = backend_class.get_doc(0)
            # Parse arguments
            backend_members = [x["name"] for x in _opts.get("members", [])]
            backend_lists = [x["name"] for x in _opts.get("lists", [])]

            tool = backend_class.__name__.lower()
            tool_options = self.edam["tool_options"][tool]

        for key, value in sorted(parsed_args.items()):
            if value is None:
                pass
            elif key in backend_members:
                tool_options[key] = value
            elif key in backend_lists:
                if not key in tool_options:
                    tool_options[key] = []
                tool_options[key] += value.split(" ")
            elif key in self.edam["parameters"]:
                _param = self.edam["parameters"][key]
                _param["default"] = value
            else:
                raise RuntimeError("Unknown parameter " + key)

    def _parse_flow_options(self, backend_class, backendargs, edam):
        available_flow_options = backend_class.get_flow_options()

        # First we check which flow options that are set in the EDAM.
        # edam["flow_options"] contain both flow and tool options, so
        # we only pick the former here
        flow_options = {}
        for k, v in edam["flow_options"].items():
            if k in available_flow_options:
                flow_options[k] = v

        # Next we build a parser and use it to parse the command-line
        progname = "fusesoc run {}".format(edam["name"])
        parser = argparse.ArgumentParser(
            prog=progname, conflict_handler="resolve", add_help=False
        )
        backend_args = parser.add_argument_group("Flow options")
        typedict = {
            "bool": {"type": str2bool, "nargs": "?", "const": True},
            "file": {"type": str, "nargs": 1, "action": FileAction},
            "int": {"type": int, "nargs": 1},
            "str": {"type": str, "nargs": 1},
            "real": {"type": float, "nargs": 1},
        }
        for k, v in available_flow_options.items():
            backend_args.add_argument(
                "--" + k,
                help=v["desc"],
                **typedict[v["type"]],
            )

        # Parse known args (i.e. only flow options) from the command-line
        parsed_args = parser.parse_known_args(backendargs)[0]

        # Clean up parsed arguments object and convert to dict
        parsed_args_dict = {}
        for key, value in sorted(vars(parsed_args).items()):
            # Remove arguments with value None, i.e. arguments not encountered
            # on the command line
            if value is None:
                continue
            _value = value[0] if type(value) == list else value

            # If flow option is a list, we split up the parsed string
            if "list" in available_flow_options[key]:
                _value = _value.split(" ")
            parsed_args_dict[key] = _value

        # Add parsed args to the ones from the EDAM
        merge_dict(flow_options, parsed_args_dict)

        return flow_options

    def parse_args(self, backend_class, backendargs):
        # First we need to see which flow options are set,
        # in order to know which tool options that are relevant
        # for this configuration of the flow
        if hasattr(backend_class, "get_flow_options"):
            self.activated_flow_options = self._parse_flow_options(
                backend_class, backendargs, self.edam
            )

        parser = self._build_parser(backend_class, self.edam)
        parsed_args = parser.parse_args(backendargs)

        args_dict = {}
        for key, value in sorted(vars(parsed_args).items()):
            if value is None:
                continue
            _value = value[0] if type(value) == list else value
            args_dict[key] = _value

        self.add_parsed_args(backend_class, args_dict)

    def to_yaml(self, edam_file):
        pathlib.Path(edam_file).parent.mkdir(parents=True, exist_ok=True)
        return utils.yaml_fwrite(edam_file, self.edam)


from fusesoc.core import Core
from fusesoc.utils import Launcher


class Ttptttg:
    def __init__(self, ttptttg, core, generators, gen_root, resolve_env_vars=False, core_list=None, toplevel=None):
        generator_name = ttptttg["generator"]
        if not generator_name in generators:
            raise RuntimeError(
                "Could not find generator '{}' requested by {}".format(
                    generator_name, core.name
                )
            )
        self.core = core
        self.generator = generators[generator_name]
        self.name = ttptttg["name"]
        self.pos = ttptttg["pos"]
        self.gen_name = generator_name
        self.gen_root = gen_root
        self.resolve_env_vars = resolve_env_vars
        parameters = ttptttg["config"]

        vlnv_str = ":".join(
            [
                core.name.vendor,
                core.name.library,
                core.name.name + "-" + self.name,
                core.name.version,
            ]
        )
        self.vlnv = Vlnv(vlnv_str)

        self.generator_input = {
            "files_root": os.path.abspath(core.files_root),
            "gapi": "1.0",
            "parameters": parameters,
            "vlnv": vlnv_str,
            "cores": core_list,
            "toplevel": toplevel,
        }

    def _sha256_input_yaml_hexdigest(self):
        return hashlib.sha256(
            utils.yaml_dump(self.generator_input).encode()
        ).hexdigest()

    def _sha256_file_input_hexdigest(self):
        input_files = []
        logger.debug(
            "Configured file_input_parameters: "
            + self.generator["file_input_parameters"]
        )
        for param in self.generator["file_input_parameters"].split():
            try:
                input_files.append(self.generator_input["parameters"][param])
            except KeyError:
                logger.debug(
                    f"Parameter {param} does not exist in parameters. File input will not be included in file input hash calculation."
                )

        logger.debug("Found input files: " + str(input_files))

        hash = hashlib.sha256()

        for f in input_files:
            abs_f = os.path.join(self.generator_input["files_root"], f)
            try:
                hash.update(pathlib.Path(abs_f).read_bytes())
            except Exception as e:
                raise RuntimeError("Unable to hash file: " + str(e))

        return hash.hexdigest()

    def _fwrite_hash(self, hashfile, data):
        with open(hashfile, "w") as f:
            f.write(data)

    def _fread_hash(self, hashfile):
        data = ""
        with open(hashfile) as f:
            data = f.read()

        return data

    def _run(self, generator_cwd):
        logger.info("Generating " + str(self.vlnv))

        generator_input_file = os.path.join(generator_cwd, self.name + "_input.yml")

        pathlib.Path(generator_cwd).mkdir(parents=True, exist_ok=True)
        utils.yaml_fwrite(generator_input_file, self.generator_input)

        args = [
            os.path.join(
                os.path.abspath(self.generator["root"]), self.generator["command"]
            ),
            os.path.abspath(generator_input_file),
        ]

        if "interpreter" in self.generator:
            interp = self.generator["interpreter"]
            interppath = shutil.which(interp)
            if not interppath:
                raise RuntimeError(
                    f"Could not find generator interpreter '{interp}' using shutil.which.\n"
                    f"Interpreter requested by generator {self.gen_name}, requested by core {self.core}.\n"
                )
            args[0:0] = [interppath]

        Launcher(args[0], args[1:], cwd=generator_cwd).run()

    def is_input_cacheable(self):
        return (
            "cache_type" in self.generator and self.generator["cache_type"] == "input"
        )

    def is_generator_cacheable(self):
        return (
            "cache_type" in self.generator
            and self.generator["cache_type"] == "generator"
        )

    def generate(self):
        """Run a parametrized generator

        Returns:
            Libary: A Library with the generated files
        """

        hexdigest = self._sha256_input_yaml_hexdigest()

        logger.debug("Generator input yaml hash: " + hexdigest)

        generator_cwd = os.path.join(
            self.gen_root or self.core.cache_root,
            "generator_cache",
            self.vlnv.sanitized_name + "-" + hexdigest,
        )

        if os.path.lexists(generator_cwd) and not os.path.isdir(generator_cwd):
            raise RuntimeError(
                "Unable to create generator working directory since it already exists and is not a directory: "
                + generator_cwd
                + "\n"
                + "Remove it manually or run 'fusesoc gen clean'"
            )

        if self.is_input_cacheable():
            # Input cache enabled. Check if cached output already exists in generator_cwd.
            logger.debug("Input cache enabled.")

            # If file_input_parameters has been configured in the generator
            # parameters will be iterated to look for files to add to the
            # input files hash calculation.
            if "file_input_parameters" in self.generator:
                file_input_hash = self._sha256_file_input_hexdigest()

                logger.debug("Generator file input hash: " + file_input_hash)

                hashfile = os.path.join(generator_cwd, ".fusesoc_file_input_hash")

                rerun = False

                if os.path.isfile(hashfile):
                    cached_hash = self._fread_hash(hashfile)
                    logger.debug("Cached file input hash: " + cached_hash)

                    if not file_input_hash == cached_hash:
                        logger.debug("File input has changed.")
                        rerun = True
                    else:
                        logger.info("Found cached output for " + str(self.vlnv))

                else:
                    logger.debug("File input hash file does not exist: " + hashfile)
                    rerun = True

                if rerun:
                    shutil.rmtree(generator_cwd, ignore_errors=True)
                    self._run(generator_cwd)
                    self._fwrite_hash(hashfile, file_input_hash)

            elif os.path.isdir(generator_cwd):
                logger.info("Found cached output for " + str(self.vlnv))
            else:
                # No directory found. Run generator.
                self._run(generator_cwd)

        elif self.is_generator_cacheable():
            # Generator cache enabled. Call the generator and let it
            # decide if the old output still is valid.
            logger.debug("Generator cache enabled.")
            self._run(generator_cwd)
        else:
            # No caching enabled. Try to remove directory if it already exists.
            # This could happen if a generator that has been configured with
            # caching is changed to no caching.
            logger.debug("Generator cache is not enabled.")
            shutil.rmtree(generator_cwd, ignore_errors=True)
            self._run(generator_cwd)

        library_name = "generated-" + self.vlnv.sanitized_name
        return Library(name=library_name, location=generator_cwd)
