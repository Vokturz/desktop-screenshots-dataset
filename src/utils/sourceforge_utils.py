from typing import Any, cast
import requests
from bs4 import BeautifulSoup

from src.model.model import Metadata

# pyright: reportInvalidTypeArguments=false
# pyright: reportAssignmentType=false
# pyright: reportArgumentType=false
# pyright: reportUndefinedVariable=false


def get_project_metadata(project_name: str, token: str | None = None) -> dict[str, Any]:
    """
    Fetch metadata for a given SourceForge project via the REST endpoint.
    Returns a dict or None if not found.
    """
    url = f"https://sourceforge.net/rest/p/{project_name}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{project_name}: {e} ({resp.status_code})")
    except ValueError:
        raise ValueError(
            f"No JSON response for project {project_name}, raw text: {resp.text[:200]}"
        )


def fetch_directory_page(page: int = 1):
    url = f"https://sourceforge.net/directory/?sort=popular&page={page}"
    r = requests.get(url)
    r.raise_for_status()
    return r.text


def fetch_project_page(slug: str):
    url = f"https://sourceforge.net/projects/{slug}"
    r = requests.get(url)
    r.raise_for_status()
    return r.text


def parse_project_slugs(html: str):
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for a in soup.select("a.project-icon"):
        href = str(a.get("href"))
        if href and href.startswith("/projects/"):
            slug = href.split("/projects/")[1].strip("/")
            links.append(slug)
    return links


def parse_metadata(metadata: dict[str, Any]) -> Metadata:
    """
    Parses metadata from a SourceForge project page.

    Args:
        metadata: The metadata dictionary as returned by the API.

    Returns:
        A Metadata object containing the parsed information.
    """
    metadata_parsed = Metadata(
        slug=metadata["shortname"],
        title=metadata["name"],
        url=metadata["url"],
        registration_date=metadata["creation_date"],
        description=metadata["short_description"],
        screenshots=[],
        topics=[],
    )
    metadata_parsed.topics = [
        topic["shortname"].lower() for topic in metadata["categories"].get("topic", [])
    ]
    metadata_parsed.screenshots = [
        screenshot["url"] for screenshot in metadata.get("screenshots", [])
    ]

    return metadata_parsed


def parse_project_page(html_content: str) -> Metadata:
    """
    Parses the HTML of a SourceForge project page to extract key details.

    Args:
        html_content: The HTML content of the page as a string.

    Returns:
        A Metadata object containing the parsed information.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    metadata = Metadata()

    # 1. Extract Project Title
    title_tag = soup.find("h1", itemprop="name")
    if title_tag:
        metadata.title = title_tag.get_text(strip=True)

    # 2. Extract date
    date_section = soup.select_one(
        'section.project-info:has(h4:-soup-contains("Registered")) section.content'
    )
    if date_section:
        metadata.registration_date = date_section.get_text(strip=True)

    # 2. Extract Project URL and Slug from the canonical link
    canonical_link = soup.find("link", rel="canonical")
    if canonical_link and "href" in canonical_link.attrs:
        url = str(canonical_link["href"]).strip()
        metadata.url = url  # <-- Store the full URL
        if "/projects/" in url:
            metadata.slug = url.split("/projects/")[1].strip("/")

    # 3. Extract the description
    description_tag = soup.find("p", itemprop="description")
    if description_tag:
        metadata.description = description_tag.get_text(strip=True)

    # 4. Extract screenshot links
    screenshot_links = soup.select(
        'section.screenshots-section a.gallery[data-featherlight="image"]'
    )
    for link in screenshot_links:
        href = link.get("href")
        if href:
            full_url = f"https:{href}"
            metadata.screenshots.append(full_url)

    # 5. Extract topic slugs
    categories_heading = soup.find("h3", string="Categories")  # pyright: ignore
    if categories_heading:
        parent_container = categories_heading.parent  # pyright: ignore
        topic_links = parent_container.find_all("a")  # pyright: ignore
        for link in topic_links:  # pyright: ignore
            href = cast(str, link.get("href"))  # pyright: ignore
            if href and href.startswith("/directory/"):
                slug = href.strip("/").split("/")[-1]
                metadata.topics.append(slug)

    return metadata


def get_mirror_metadata(slug: str) -> Metadata:
    html = fetch_project_page(slug)
    metadata = parse_project_page(html)

    if not metadata.title and not metadata.description:
        raise ValueError(f"Metadata for {slug} is incomplete")
    return metadata
