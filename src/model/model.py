from dataclasses import dataclass, asdict, field
import json
from typing import Any

# pyright: reportUnknownVariableType=false


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


def filter_none_values(data: Any):  # pyright: ignore
    """Recursively filter out None values from dictionaries and lists."""
    if isinstance(data, dict):
        return {k: filter_none_values(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [filter_none_values(item) for item in data if item is not None]
    else:
        return data


@dataclass
class ImageUrl:
    url: str = ""


@dataclass
class Content:
    type: str = ""
    text: str | None = None
    image_url: ImageUrl | None = None

    @classmethod
    def text_type(cls, text: str):
        return cls(type="text", text=text)

    @classmethod
    def image_url_type(cls, url: str):
        image_url = ImageUrl(url=url)
        return cls(type="image_url", image_url=image_url)


@dataclass
class Message:
    role: str = ""
    content: list[Content] = field(default_factory=list)

    def to_dict(self):  # pyright: ignore
        return filter_none_values(asdict(self))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)

    @classmethod
    def user(cls, text: str, image_url: str | None = None):
        contents = [Content.text_type(text)]
        if image_url:
            image_content = Content.image_url_type(image_url)
            contents.append(image_content)
        return cls(role="user", content=contents)

    @classmethod
    def assistant(cls, text: str):
        content = Content.text_type(text)
        return cls(role="assistant", content=[content])

    @classmethod
    def system(cls, text: str):
        content = Content.text_type(text)
        return cls(role="system", content=[content])


@dataclass
class VLMConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 1.5
    stream: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class ImageAnalysis:
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    category: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        description = data.get("description", "")
        keywords = data.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]
        category = data.get("category", "")
        return cls(description=description, keywords=keywords, category=category)
