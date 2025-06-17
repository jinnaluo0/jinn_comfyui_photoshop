
import time
from nodes import KSampler,KSamplerAdvanced
@classmethod
def IS_CHANGED(cls, **kwargs):
    return time.time()
KSampler.IS_CHANGED = IS_CHANGED
KSamplerAdvanced.IS_CHANGED = IS_CHANGED