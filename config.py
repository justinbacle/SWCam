import os
import sys
import PySide2

if sys.platform == "linux":
    CTI_FILEPATH = "/opt/spinnaker/lib/flir-gentl/FLIR_GenTL.cti"

    # PySide default configuration is messed up in linux
    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

else:
    CTI_FILEPATH = "C:/Program Files/FLIR Systems/Spinnaker/bin64/vs2015/FLIR_GenTL_v140.cti"

MAKEDNG_PATH = os.getcwd() + "/makeDNG/makeDNG.exe"
