import base64
import io
import os
from typing import Any
import re
import requests
import matplotlib.pyplot as plt

from src.model.model import ImageAnalysis, VLMConfig, Message
from src.utils.system_prompts import OCR, SCREENSHOT_CLASSIFIER, SCREENSHOT_VALIDITY
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
    timeout: float | None = None,
):
    """Send a request to the Vision Language Model API.

    Makes an HTTP POST request to the configured VLM endpoint with the provided
    messages and configuration. Uses environment variables for API configuration.

    Args:
        messages: List of Message objects to send to the API
        config: Optional VLM configuration. If None, uses default VLMConfig()
        timeout: Optional timeout in seconds for the request

    Returns:
        str: The assistant's response content from the API

    Raises:
        requests.HTTPError: If the API request fails
        requests.Timeout: If the request times out

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

    data: dict[str, Any] = {
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
        **config.to_dict(),
    }
    response = requests.post(url, headers=headers, json=data, timeout=timeout)
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
    return not assistant_message.lower().strip().endswith("invalid")


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
    from_src: bool = False,
    attempt: int = 1,
):
    """Analyze a screenshot using VLM to extract description, keywords, and category.

    Performs comprehensive analysis of a desktop screenshot using a Vision Language
    Model. Uses few-shot learning with predefined examples to ensure consistent
    output format. Extracts structured information in XML tags.

    The function validates that the extracted category is one of the predefined valid
    categories. If not, it retries up to 3 times with additional instructions to
    select a valid category. After 3 failed attempts, it defaults to 'other'.

    Args:
        example: Dictionary containing:
            - 'image': PIL Image object of the screenshot
            - 'slug': Process name string
            - 'title': Application window title
        print_assistant_message: Whether to print the full VLM response
        custom_system_prompt: Optional custom system prompt to override default
        from_src: Whether to use src-relative paths for example images
        attempt: Current attempt number (1-3). Used internally for retry logic.

    Returns:
        ImageAnalysis: Object containing extracted description, keywords, and category

    Raises:
        StopIteration: If required XML tags are not found in the VLM response
        AttributeError: If regex match fails to find expected content

    Note:
        The function expects the VLM to respond with content wrapped in
        <description>, <keywords>, and <category> XML tags. Category validation
        ensures the extracted category matches one of the predefined options.
    """
    # Define valid categories from system prompt
    valid_categories = {
        "code editor",
        "terminal",
        "document editor",
        "spreadsheets",
        "database tools",
        "email app",
        "chat/messaging",
        "video conferencing",
        "file manager",
        "music streaming",
        "video streaming",
        "social media",
        "online shopping",
        "research/browsing",
        "game",
        "media editing",
        "system utilities",
        "productivity/project tools",
        "finance/accounting",
        "other",
    }

    user_text = f'This application whose process name is "{example["slug"]}" has the following title: "{example["title"]}".'
    user_text += " Describe only what you can see in this screenshot without making assumptions about user intentions or goals. Don't forget to use <description>, <keywords>, and <category> tags separately."

    # Add category validation instruction on retry attempts
    if attempt > 1:
        categories_list = ", ".join(sorted(valid_categories))
        user_text += f"\n\nIMPORTANT: The category MUST be exactly one of these options: {categories_list}. Please ensure you select the most appropriate category from this list."

    example_1 = [
        "src/data/examples/code-editor.png"
        if from_src
        else "../data/examples/code-editor.png",
        'This application whose process name is "dev.zed.Zed" has the following title: "loyca-ai - sql.rs"',
        """<description>The screenshot shows a code editor window from the application "Zed" with the project "loyca-ai" open. The left sidebar displays the project’s folder structure. Inside `src/tauri/src/sql/`, the file `sql.rs` is currently open in the main editor area. The code is written in Rust, defining functions such as `init` and `setup_database`. The code deals with initializing a SQLite database connection using `OnceCell` and `Mutex`. There are imports for `once_cell`, `rusqlite`, `tauri`, and others. The cursor is located in the main code editing area, near the end of the `init` function implementation.</description>
        <keywords>Zed, code editor, Rust, sql.rs, OnceCell, rusqlite, SQLite, database, src-tauri, backend</keywords>
    <category>code editor</category>""",
    ]
    example_2 = [
        "src/data/examples/reddit.png" if from_src else "../data/examples/reddit.png",
        'This application whose process name is "app.zen_browser.zen" has the following title: "Reddit - The heart of the internet — Zen Browser".',
        """"<description>The screen displays the Reddit website within a web browser. The user is viewing a feed, with the top post from the subreddit "r/PeterExplainsTheJoke" titled "What game??". This post contains a video of a Chuck E. Cheese building on fire, with a large plume of smoke and a fire truck ladder visible. Text overlaid on the video says, "Somebody beat the game". Below this, a second post from "r/LocalLLaMA" titled "Local reasoning model" is partially visible.</description>
    <keywords>Reddit, social media, meme, r/PeterExplainsTheJoke, r/LocalLLaMA, AI models, LLM, browsing</keywords>
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

    # Validate category and retry if invalid
    extracted_category = result["category"]
    if extracted_category not in valid_categories:
        if attempt <= 3:
            print(
                f"Invalid category '{extracted_category}' on attempt {attempt}/3. Retrying..."
            )
            return do_image_analysis(
                example=example,
                print_assistant_message=print_assistant_message,
                custom_system_prompt=custom_system_prompt,
                from_src=from_src,
                attempt=attempt + 1,
            )
        else:
            print(
                f"Failed to get valid category after 3 attempts. Got: '{extracted_category}'. Using 'other' as fallback."
            )
            result["category"] = "other"

    return ImageAnalysis.from_dict(result)


def do_ocr(
    example: dict[str, Any],
    image_analysis: ImageAnalysis | None = None,
    print_assistant_message: bool = False,
    attempt: int = 1,
    custom_system_prompt: str | None = None,
) -> str:
    """Perform Optical Character Recognition on a screenshot using VLM.

    Uses a Vision Language Model to extract text content from images. Can handle
    both general OCR tasks and specific image descriptions. Special handling for
    markdown code blocks in the response. Implements retry logic with up to 3 attempts.

    Args:
        example: Dictionary containing:
            - 'image': PIL Image object to perform OCR on
            - 'title': Application title (used if no image_description provided)
        image_analysis: Custom ImageAnalysis object for OCR task. If None,
            defaults to OCR request for the application title
        print_assistant_message: Whether to print the full VLM response
        attempt: Current attempt number (1-3). Used internally for retry logic.
        custom_system_prompt: Optional custom system prompt to override default

    Returns:
        str: Extracted text content. If VLM returns markdown code blocks,
            returns only the content within the blocks. Otherwise returns
            the full response.

    Note:
        Attempts 1-2 use a 10-second timeout. Attempt 3 has no timeout.
        On attempts 2-3, adds a note to avoid repetitive content.
        If the VLM response contains a markdown code block
        (```markdown...```), only the content inside the block is returned.
        Other code block types and plain text responses are returned as-is.
    """
    user_text = (
        image_analysis.description
        if image_analysis
        else f'Do OCR for "{example["title"]}".'
    )
    if attempt > 2:
        user_text += "\n\nIMPORTANT: I don't need the whole content. Focus on main elements only and REPLACE REPETITIVE TEXT with `[...]`."
    elif attempt > 1:
        user_text += "\n\nIMPORTANT: I don't need the whole content. Avoid background text and REPLACE REPETITIVE TEXT with `[...]`."
    else:
        user_text += (
            "\n\nIMPORTANT: Remember, DO NOT MAKE UP content and AVOID REPETITIVE TEXT."
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

    # Attempts 1-2 with timeout, attempt 3 without timeout
    if attempt <= 2:
        try:
            assistant_message = do_vlm_request(messages, timeout=10.0)
        except requests.Timeout:
            if attempt < 3:
                print(
                    f"OCR: Timeout occurred for {example['title']} (attempt {attempt}/3)"
                )
                return do_ocr(
                    example=example,
                    image_analysis=image_analysis,
                    print_assistant_message=print_assistant_message,
                    attempt=attempt + 1,
                    custom_system_prompt=custom_system_prompt,
                )
            else:
                # Final attempt failed, re-raise the timeout
                raise
    else:
        assistant_message = do_vlm_request(messages, timeout=30.0)

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
