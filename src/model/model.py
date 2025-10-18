from dataclasses import dataclass, asdict, field
import json
from typing import Any


@dataclass
class Metadata:
    slug: str = ""
    url: str = ""
    title: str = ""
    registration_date: str = ""
    description: str = ""
    is_mirror: bool = False
    screenshots: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)
