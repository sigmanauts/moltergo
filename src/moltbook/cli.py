import argparse
import json
from pathlib import Path
from typing import Any

from .moltbook_client import MoltbookAuthError, MoltbookClient


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def cmd_me(_: argparse.Namespace) -> None:
    """Show information about the current Moltbook agent."""
    client = MoltbookClient()
    me = client.get_me()
    print_json(me)


def cmd_post(args: argparse.Namespace) -> None:
    """Create a text or link post on Moltbook.

    Examples:

        python -m moltbook.cli post \
          --submolt general \
          --title "Ergo for AI agents" \
          --content "Why Ergo is programmable money that fits AI needs"

        python -m moltbook.cli post \
          --submolt general \
          --title "Ergo docs" \
          --url "https://ergoplatform.org/en/"
    """

    if not args.content and not args.url:
        raise SystemExit("You must provide either --content or --url (or both).")

    client = MoltbookClient()
    resp = client.create_post(
        submolt=args.submolt,
        title=args.title,
        content=args.content,
        url=args.url,
    )
    print_json(resp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple Moltbook CLI for posting and inspecting your agent.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # me
    p_me = subparsers.add_parser("me", help="Show current agent profile")
    p_me.set_defaults(func=cmd_me)

    # post
    p_post = subparsers.add_parser("post", help="Create a post on Moltbook")
    p_post.add_argument("--submolt", required=True, help="Target submolt, e.g. 'general' or 'ergo'")
    p_post.add_argument("--title", required=True, help="Title of the post")
    p_post.add_argument("--content", help="Text content of the post")
    p_post.add_argument("--url", help="Optional URL for link posts")
    p_post.set_defaults(func=cmd_post)

    return parser


def main() -> None:
    try:
        parser = build_parser()
        args = parser.parse_args()
        args.func(args)
    except MoltbookAuthError as e:
        raise SystemExit(str(e))
    except Exception as e:
        # Catch-all to avoid noisy tracebacks for common runtime issues
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()
