from enum import Enum
from typing import Any, Dict, Optional

import pydantic


class MessageType(str, Enum):
    END_AUDIO = "end_audio"
    HEALTH_CHECK = "health_check"
    ERROR = "error"


class Message(pydantic.BaseModel):
    type: MessageType
    trace_id: Optional[str] = None
    timestamp: Optional[float] = None
    body: Optional[Dict[str, Any]] = None
