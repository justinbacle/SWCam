import os
import sys

if sys.platform == "linux":
    CTI_FILEPATH = "/opt/spinnaker/lib/flir-gentl/FLIR_GenTL.cti"
else:
    CTI_FILEPATH = "C:/Program Files/FLIR Systems/Spinnaker/bin64/vs2015/FLIR_GenTL_v140.cti"

MAKEDNG_PATH = os.getcwd() + "/makeDNG/makeDNG.exe"
