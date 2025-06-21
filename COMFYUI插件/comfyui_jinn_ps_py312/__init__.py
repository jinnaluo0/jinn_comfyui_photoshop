#raise Exception()
#from . import jinn_nodes
from . import jinn_server
NODE_CLASS_MAPPINGS={}
NODE_DISPLAY_NAME_MAPPINGS={}
try:
    pass
    #NODE_CLASS_MAPPINGS.update(jinn_nodes.NODE_CLASS_MAPPINGS)
    #NODE_DISPLAY_NAME_MAPPINGS.update(jinn_nodes.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    pass 

#WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]#, "WEB_DIRECTORY"

