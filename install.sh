#!/usr/bin/env bash

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASHRC="${HOME}/.bashrc"

chmod +x "${REPO_DIR}/bill"

PATH_LINE="export PATH=\"${REPO_DIR}:\$PATH\""

if grep -qF "$PATH_LINE" "$BASHRC" 2>/dev/null; then
    echo "PATH already configured in ${BASHRC}"
else
    {
        echo ""
        echo "# UC3M C3 Expense Calculator"
        echo "$PATH_LINE"
    } >> "$BASHRC"

    echo "Added ${REPO_DIR} to PATH"
fi

echo
echo "Installation complete."
echo "Run:    source ~/.bashrc"
echo "Then:   bill"
