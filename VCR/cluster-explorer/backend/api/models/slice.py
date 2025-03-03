from enum import Enum
from typing import Optional

from pydantic import BaseModel

class ErrorTypes(str, Enum):
    IOU_FP = "iou_fp"
    CLASS_FP = "class_fp"

class VisRequest(BaseModel):
    file: str
    slice: str
    limit: int

    # filtering options
    label: Optional[int] = None
    count: Optional[bool] = False
    crowding: Optional[bool] = False
    bbox_area: Optional[bool] = False
    qcut: Optional[bool] = False
    cut: Optional[bool] = False

    def __hash__(self):
        return hash((
            self.file,
            self.slice,
            self.limit,
            self.label,
            self.count,
            self.crowding,
            self.bbox_area,
            self.qcut,
            self.cut,
        ))

class LookupRequest(BaseModel):
    file: str
    slice: str
    # limit: int

    # filtering options
    # label: Optional[int] = None
    # crowding: Optional[bool] = False
    # bbox_area: Optional[bool] = False
    # qcut: Optional[bool] = False
    # cut: Optional[bool] = False

class MineRequest(BaseModel):
    max_combo: int = 3
    limit: Optional[int] = 50000
    support: float = 0.01
    label: Optional[str] = None
    top_k: int = 30
    error_type: Optional[ErrorTypes] = ErrorTypes.IOU_FP
    file: str = None
    label: Optional[int] = None

    # filtering options
    count: Optional[bool] = False
    crowding: Optional[bool] = False
    bbox_area: Optional[bool] = False
    qcut: Optional[bool] = False
    cut: Optional[bool] = False
    dedup: Optional[str] = None

    class Config:
        use_enum_values = True

    def __hash__(self):
        return hash((
            self.max_combo,
            self.limit,
            self.support,
            self.label,
            self.top_k,
            self.file,
            self.count,
            self.crowding,
            self.bbox_area,
            self.qcut,
            self.cut,
            self.dedup,
        ))