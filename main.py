import logging
import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq


logger = logging.getLogger(__name__)


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> int:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if not api_key:
        logger.error("Missing GROQ_API_KEY environment variable")
        return 1

    try:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            timeout=45,
            max_retries=1,
        )
        response = llm.invoke(
            "Explain what an HTTP 500 error means in one sentence."
        )
    except Exception:
        logger.exception("Groq request failed")
        return 1

    print(response.content)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
