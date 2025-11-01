import base64
import io
import os
from typing import Any
import re
import requests
import matplotlib.pyplot as plt

from model.model import ImageAnalysis, VLMConfig, Message
from utils.system_prompts import OCR, SCREENSHOT_CLASSIFIER, SCREENSHOT_VALIDITY

# pyright: reportUnknownMemberType=false


def display_image(example: dict[str, Any], figsize: tuple[int, int] = (12, 10)):
    """Display an image from a dataset example dictionary using matplotlib.

    Args:
        example: Dictionary containing 'image' (PIL Image) and 'slug' (str) keys
        figsize: Figure size as (width, height) in inches. Defaults to (12, 10)

    Note:
        This function will show the plot using plt.show(), which may block execution
        in some environments until the plot window is closed.
    """
    plt.figure(figsize=figsize)
    plt.imshow(example["image"])
    plt.title(example["slug"])
    plt.axis("off")
    plt.show()


def do_vlm_request(
    messages: list[Message],
    config: VLMConfig | None = None,
):
    """Send a request to the Vision Language Model API.

    Makes an HTTP POST request to the configured VLM endpoint with the provided
    messages and configuration. Uses environment variables for API configuration.

    Args:
        messages: List of Message objects to send to the API
        config: Optional VLM configuration. If None, uses default VLMConfig()

    Returns:
        str: The assistant's response content from the API

    Raises:
        requests.HTTPError: If the API request fails

    Environment Variables:
        MODEL: Model name to use (default: "Qwen/Qwen3-VL-30B-A3B-Instruct")
        BASE_URL: API base URL (default: "http://localhost:8000")
        API_KEY: API key for authentication (default: "None")
    """
    model = os.getenv("MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "None")

    if config is None:
        config = VLMConfig()

    url = f"{base_url}/v1/chat/completions"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    data = {  # pyright: ignore
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
        **config.to_dict(),
    }
    response = requests.post(url, headers=headers, json=data)  # pyright: ignore
    response.raise_for_status()

    resp = response.json()
    assistant_message = resp["choices"][0]["message"]["content"]
    return assistant_message


def check_screenshot_validity(
    example: dict[str, Any],
    print_assistant_message: bool = False,
    custom_system_prompt: str | None = None,
):
    """Check if a screenshot is valid for the specified application.

    Uses a VLM to determine if the provided screenshot actually shows the
    application described in the example metadata. Converts the image to
    base64 format and sends it to the VLM with contextual information.

    Args:
        example: Dictionary containing:
            - 'image': PIL Image object of the screenshot
            - 'title': Application title string
            - 'description': Optional application description from sourceforge
        print_assistant_message: Whether to print the full VLM response
        custom_system_prompt: Optional custom system prompt to override default

    Returns:
        bool: True if the screenshot is deemed valid, False otherwise

    Note:
        Validity is determined by checking if the VLM response starts with "valid"
        (case-insensitive).
    """
    user_text = f'Is this image from the application "{example["title"]}" valid?.'
    if example["description"]:
        user_text += f" This is the description from sourceforge:\n{example['description'].split('\n')[0]}\n"

    img = example["image"]
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    system_prompt = custom_system_prompt or SCREENSHOT_VALIDITY

    messages = [
        Message.system(system_prompt),
        Message.user(text=user_text, image_url=f"data:image/png;base64,{image_base64}"),
    ]

    assistant_message = do_vlm_request(messages)

    if print_assistant_message:
        print(assistant_message)
    return assistant_message.lower().strip().startswith("valid")


def create_example(image_path: str, user_text: str, assistant_text: str):
    """Create a conversation example for few-shot learning.

    Loads an image from disk, converts it to base64, and creates a user-assistant
    message pair for use in few-shot prompting scenarios.

    Args:
        image_path: Path to the image file on disk
        user_text: Text content for the user message
        assistant_text: Text content for the assistant response

    Returns:
        list[Message]: List containing user and assistant Message objects

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If the image file cannot be read
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    user_message = Message.user(
        text=user_text, image_url=f"data:image/png;base64,{image_base64}"
    )
    assistant_message = Message.assistant(text=assistant_text)

    return [user_message, assistant_message]


def do_image_analysis(
    example: dict[str, Any],
    print_assistant_message: bool = False,
    custom_system_prompt: str | None = None,
):
    """Analyze a screenshot using VLM to extract description, keywords, and category.

    Performs comprehensive analysis of a desktop screenshot using a Vision Language
    Model. Uses few-shot learning with predefined examples to ensure consistent
    output format. Extracts structured information in XML tags.

    Args:
        example: Dictionary containing:
            - 'image': PIL Image object of the screenshot
            - 'slug': Process name string
            - 'title': Application window title
        print_assistant_message: Whether to print the full VLM response
        custom_system_prompt: Optional custom system prompt to override default

    Returns:
        ImageAnalysis: Object containing extracted description, keywords, and category

    Raises:
        StopIteration: If required XML tags are not found in the VLM response
        AttributeError: If regex match fails to find expected content

    Note:
        The function expects the VLM to respond with content wrapped in
        <description>, <keywords>, and <category> XML tags.
    """
    user_text = f'This application whose process name is "{example["slug"]}" has the following title: "{example["title"]}".'
    user_text += " Describe only what you can see in this screenshot without making assumptions about user intentions or goals. Don't forget to use <description>, <keywords>, and <category> tags separately."

    example_1 = [
        "../data/examples/code-editor.png",
        'This application whose process name is "dev.zed.Zed" has the following title: "loyca-ai - sql.rs"',
        """<description>
    A Rust file, `sql.rs`, is open in the "loyca-ai" Tauri project. The code implements a thread-safe, lazily-initialized singleton pattern for a SQLite database connection using the `once_cell` and `Mutex` crates. The `init` function retrieves the application's data directory via the Tauri API to create or open an `sqlite.db` file with `rusqlite`.
    </description>
    <keywords>Rust, Tauri, SQLite, rusqlite, database connection, singleton pattern, thread safety, OnceCell, backend development</keywords>
    <category>code editor</category>""",
    ]
    example_2 = [
        "../data/examples/reddit.png",
        'This application whose process name is "app.zen_browser.zen" has the following title: "Reddit - The heart of the internet — Zen Browser".',
        """"<description>
    The user is browsing their Reddit feed. Two posts are visible: one from the subreddit r/PeterExplainsTheJoke titled "What game??", showing an image of a Chuck E. Cheese building with smoke coming from it and the caption "Somebody beat the game". Below it is a post from r/LocalLLaMA titled "Local reasoning model", asking for recommendations for non-Chinese AI reasoning models.
    </description>
    <keywords>Reddit, social media, meme, r/PeterExplainsTheJoke, r/LocalLLaMA, AI models, LLM, reasoning model, browsing</keywords>
    <category>social media</category>""",
    ]

    img = example["image"]
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    few_shot_examples = create_example(*example_1) + create_example(*example_2)
    system_prompt = custom_system_prompt or SCREENSHOT_CLASSIFIER

    messages = [
        Message.system(system_prompt),
        *few_shot_examples,
        Message.user(text=user_text, image_url=f"data:image/png;base64,{image_base64}"),
    ]

    assistant_message = do_vlm_request(messages)

    if print_assistant_message:
        print(assistant_message)

    result: dict[str, Any] = dict()
    for xml_tag in ["description", "keywords", "category"]:
        content = (
            next(
                re.finditer(
                    f"<{xml_tag}>(.*?)</{xml_tag}>", assistant_message, re.DOTALL
                )
            )
            .group(1)
            .strip()
        )
        result[xml_tag] = content
    return ImageAnalysis.from_dict(result)


def do_ocr(
    example: dict[str, Any],
    image_description: str = "",
    print_assistant_message: bool = False,
    custom_system_prompt: str | None = None,
):
    """Perform Optical Character Recognition on a screenshot using VLM.

    Uses a Vision Language Model to extract text content from images. Can handle
    both general OCR tasks and specific image descriptions. Special handling for
    markdown code blocks in the response.

    Args:
        example: Dictionary containing:
            - 'image': PIL Image object to perform OCR on
            - 'title': Application title (used if no image_description provided)
        image_description: Custom description/prompt for OCR task. If empty,
            defaults to OCR request for the application title
        print_assistant_message: Whether to print the full VLM response
        custom_system_prompt: Optional custom system prompt to override default

    Returns:
        str: Extracted text content. If VLM returns markdown code blocks,
            returns only the content within the blocks. Otherwise returns
            the full response.

    Note:
        If the VLM response contains a markdown code block (```markdown...```),
        only the content inside the block is returned. Other code block types
        and plain text responses are returned as-is.
    """
    user_text = (
        image_description if image_description else f'Do OCR for "{example["title"]}". '
    )

    img = example["image"]
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    system_prompt = custom_system_prompt or OCR

    messages = [
        Message.system(system_prompt),
        Message.user(text=user_text, image_url=f"data:image/png;base64,{image_base64}"),
    ]

    assistant_message = do_vlm_request(messages)

    if print_assistant_message:
        print(assistant_message)

    # Check for code blocks and handle markdown specifically
    code_block_pattern = r"```(\w+)?\n?(.*?)\n?```"
    match = re.search(code_block_pattern, assistant_message, re.DOTALL)

    if match:
        language = match.group(1)  # Could be None if no language specified
        content = match.group(2)

        # If it's markdown, return only the content
        if language and language.lower() == "markdown":
            return content.strip()

    # Return everything as-is for non-markdown code blocks or no code blocks
    return assistant_message
