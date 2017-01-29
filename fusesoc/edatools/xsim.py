import os
from .simulator import Simulator
import logging
from fusesoc.utils import Launcher

logger = logging.getLogger(__name__)

class Xsim(Simulator):

    MAKEFILE_TEMPLATE="""#Auto generated by FuseSoC
include config.mk

all: xsim.dir/$(TARGET)/xsimk

xsim.dir/$(TARGET)/xsimk:
	xelab $(TOPLEVEL) -prj $(TARGET).prj -snapshot $(TARGET) $(VLOG_DEFINES) $(VLOG_INCLUDES) $(VLOG_PARAMS) $(XELAB_OPTIONS)

run: xsim.dir/$(TARGET)/xsimk
	xsim --tclbatch run.tcl $(XSIM_OPTIONS) $(TARGET) $(EXTRA_OPTIONS)

run-gui: xsim.dir/$(TARGET)/xsimk
	xsim --gui --tclbatch run-gui.tcl $(XSIM_OPTIONS) $(TARGET) $(EXTRA_OPTIONS)
"""

    CONFIG_MK_TEMPLATE = """#Auto generated by FuseSoC
TARGET        = {target}
TOPLEVEL      = {toplevel}

VLOG_DEFINES  = {vlog_defines}
VLOG_INCLUDES = {vlog_includes}
VLOG_PARAMS   = {vlog_params}

XELAB_OPTIONS =	{xelab_options}
"""

    RUN_TCL_TEMPLATE = """#Auto generated by FuseSoC
run all
quit
"""

    RUN_GUI_TCL_TEMPLATE = """#Auto generated by FuseSoC
add_wave -radix hex /
run all
"""

    def configure(self, args):
        super(Xsim, self).configure(args)
        self._write_config_files()

        #Check if any VPI modules are present and display warning
        if len(self.vpi_modules) > 0:
            modules = [m['name'] for m in self.vpi_modules]
            logger.error('VPI modules not supported by Xsim: %s' % ', '.join(modules))

    def _write_config_files(self):
        with open(os.path.join(self.work_root, self.name+'.prj'),'w') as f:
            (src_files, self.incdirs) = self._get_fileset_files()
            for src_file in src_files:
                cmd = ""
                if src_file.file_type.startswith("verilogSource"):
                    cmd = 'verilog'
                elif src_file.file_type == 'vhdlSource-2008':
                    cmd = 'vhdl2008'
                elif src_file.file_type.startswith("vhdlSource"):
                    cmd = 'vhdl'
                elif src_file.file_type.startswith("systemVerilogSource"):
                    cmd = 'sv'
                elif src_file.file_type in ["user"]:
                    pass
                else:
                    _s = "{} has unknown file type '{}'"
                    logger.warning(_s.format(src_file.name, src_file.file_type))
                if cmd:
                    if src_file.logical_name:
                        lib = src_file.logical_name
                    else:
                        lib = 'work'
                    f.write('{} {} {}\n'.format(cmd, lib, src_file.name))

        with open(os.path.join(self.work_root, 'config.mk'), 'w') as f:
            vlog_defines  = ' '.join(['--define {}={}'.format(k,v) for k,v, in self.vlogdefine.items()])
            vlog_includes = ' '.join(['-i '+k for k in self.incdirs])
            vlog_params   = ' '.join(['--generic_top {}={}'.format(k, self._param_value_str(v)) for k,v, in self.vlogparam.items()])
            xelab_options = '--timescale 1ps/1ps --debug typical'
            if 'xsim_options' in self.tool_options:
                xelab_options += ' ' + ' '.join(self.tool_options['xsim_options'])

            f.write(self.CONFIG_MK_TEMPLATE.format(target=self.name,
                                                   toplevel=self.toplevel,
                                                   vlog_defines = vlog_defines,
                                                   vlog_includes = vlog_includes,
                                                   vlog_params   = vlog_params,
                                                   xelab_options = xelab_options))

        with open(os.path.join(self.work_root, 'run.tcl'), 'w') as f:
            f.write(self.RUN_TCL_TEMPLATE)
        with open(os.path.join(self.work_root, 'run-gui.tcl'), 'w') as f:
            f.write(self.RUN_GUI_TCL_TEMPLATE)
        with open(os.path.join(self.work_root, 'Makefile'), 'w') as f:
            f.write(self.MAKEFILE_TEMPLATE)

    def run(self, args):
        super(Xsim, self).run(args)

        args = ['run']
        # Plusargs
        if self.plusarg:
            _s = '--testplusarg {}={}'
            args.append('EXTRA_OPTIONS='+' '.join([_s.format(k, v) for k,v in self.plusarg.items()]))

        Launcher('make', args,
                 cwd = self.work_root,
                 errormsg = "Failed to run Xsim simulation").run()

        super(Xsim, self).done(args)