"""Make the VAPID key pair for the room call's meeting reminders and store it as a Modal secret.

    uv run --env-file .env python scripts/roomcall/vapid_secret.py --subject mailto:you@your.domain

Run it ONCE. The private key goes straight into the Modal secret `roomcall-vapid` (VAPID_PRIVATE_KEY,
VAPID_PUBLIC_KEY, VAPID_SUBJECT) through a temporary file only you can read, removed right after;
it is never printed, logged or written anywhere else. Only the public key is printed.

Every browser that asked for a reminder is subscribed against the public key: making a new pair
(`--rotate`) silently ends every reminder already set up, so this refuses to replace an existing
secret without it.

--subject is how push services (Apple's above all) can reach whoever runs the server: a mailto: or
https: address. The default is a placeholder: give your own (re-run with --rotate to change it,
which also replaces the keys).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PLACEHOLDER_SUBJECT = "mailto:roomcall-admin@example.com"


def make_keys() -> tuple[str, str]:
    """(private, public), base64url without padding: the raw 32-byte private scalar (what
    pywebpush's `Vapid.from_string` reads) and the 65-byte uncompressed public point (what
    browsers take as `applicationServerKey`)."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_numbers().private_value.to_bytes(32, "big")
    public = key.public_key().public_bytes(serialization.Encoding.X962,
                                           serialization.PublicFormat.UncompressedPoint)

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    return b64(private), b64(public)


def main(argv: list[str] | None = None) -> int:
    from litereality_agent.room_ops.roomcall.meetings import (
        REMINDERS_DEPLOY_FILE,
        VAPID_SECRET,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subject", default=PLACEHOLDER_SUBJECT,
                        help="mailto: or https: contact for push services (default: a placeholder)")
    parser.add_argument("--env", default="main", help="Modal environment (default: main)")
    parser.add_argument("--rotate", action="store_true",
                        help="replace an existing secret (ends every reminder already set up)")
    args = parser.parse_args(argv)
    if not args.subject.startswith(("mailto:", "https://")):
        parser.error("--subject must start with mailto: or https://")

    private, public = make_keys()
    values = {"VAPID_PRIVATE_KEY": private, "VAPID_PUBLIC_KEY": public,
              "VAPID_SUBJECT": args.subject}
    with tempfile.TemporaryDirectory(prefix="roomcall-vapid-") as folder:
        path = Path(folder) / "vapid.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(values, handle)
        command = [sys.executable, "-m", "modal", "secret", "create", VAPID_SECRET,
                   "--from-json", str(path), "--env", args.env]
        if args.rotate:
            command.append("--force")
        # Modal's own output names the secret and its keys, never their values.
        result = subprocess.run(command, check=False)
    del private, values
    if result.returncode != 0:
        print(f"\ncreating the Modal secret {VAPID_SECRET!r} failed (see above). If it already "
              "exists, keep it: pass --rotate only to replace the keys, which ends every reminder "
              "already set up.", file=sys.stderr)
        return result.returncode
    print(f"\nModal secret {VAPID_SECRET!r} created in {args.env!r}.")
    print(f"  public key   {public}")
    print(f"  subject      {args.subject}")
    if args.subject == PLACEHOLDER_SUBJECT:
        print("  (a placeholder subject: re-run with --rotate --subject mailto:<you> to set yours)")
    print(f"\nnext: uv run --env-file .env modal deploy --env {args.env} {REMINDERS_DEPLOY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
