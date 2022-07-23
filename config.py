import os
import logging
import sys
import PySide6

# ---------------------------------- Logging --------------------------------- #
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

# ---------------------------- GenTL Producer Path --------------------------- #
try:
    CTI_FILEPATH = os.getenv("GENICAM_GENTL64_PATH")
    if CTI_FILEPATH is not None:
        logging.info(f"Found GenTL producer: {CTI_FILEPATH}")
    else:
        CTI_FILEPATH = os.getenv("GENICAM_GENTL32_PATH")
        if CTI_FILEPATH is not None:
            logging.info(f"Found GenTL producer: {CTI_FILEPATH}")
        else:
            raise FileNotFoundError
except FileNotFoundError:
    if sys.platform == "linux":
        CTI_FILEPATH = "/opt/spinnaker/lib/flir-gentl/FLIR_GenTL.cti"
        # CTI_FILEPATH = "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"

    else:
        CTI_FILEPATH = "C:/Program Files/FLIR Systems/Spinnaker/bin64/vs2015/FLIR_GenTL_v140.cti"

# ------------------------------- Qt Specifics ------------------------------- #
if sys.platform == "linux":
    # PySide default configuration is messed up in linux
    dirname = os.path.dirname(PySide6.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

# --------------------------------- Make DNG --------------------------------- #
MAKEDNG_PATH = os.getcwd() + "/makeDNG/makeDNG.exe"
