# Modified for JustFit: product identity and integration.
"""Adapt the qualified vision worker lifecycle to the JustFit frame API."""
import json
from dataclasses import dataclass,replace
from . import protocol as wire
from .prepared_vision import from_profile as worker_from_profile

@dataclass(frozen=True)
class _Input:
    image_spans:tuple
    image_pixels:bytes
    image_embedding_width:int
    image_owner:object

class VisionAdapter:
    def __init__(self,worker):self.worker=worker
    def prepare(self,request,cancelled,deadline):
        frame=request.frame
        if not frame.image_spans or frame.image_embedding_width:return request
        original=_Input(frame.image_spans,frame.image_pixels,0,request.image_owner)
        result=self.worker.prepare(original,cancelled,deadline)
        if result is original:return request
        updated=replace(frame,image_spans=result.image_spans,image_pixels=result.image_pixels,
                        image_embedding_width=result.image_embedding_width,
                        flags=frame.flags|wire.RequestFlag.VISION_EMBEDDINGS)
        return replace(request,frame=updated,image_owner=result.image_owner)
    def close(self):self.worker.close()
    def snapshot(self):return self.worker.snapshot()

def from_profile(path,runtime,model_root,host_reserve):
    status=json.loads(runtime.status().json)
    if status.get("justfit_vision_extensions") != 1:
        raise ValueError("ANE vision requires the matching embedding/lease native binary")
    return VisionAdapter(worker_from_profile(path,runtime,model_root,host_reserve))
