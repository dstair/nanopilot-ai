#!/bin/bash
# Build XboxBridge.app — a real .app bundle that reads Xbox controller state
# via Apple's GameController framework and forwards to /tmp/nanopilot_xbox.sock.
#
# Required because Apple's XboxGamepad dext claims exclusive HID ownership;
# only GameController.framework (in a real bundle) can read inputs.

set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="XboxBridge"
BUILD_DIR="build"
APP_BUNDLE="${BUILD_DIR}/${APP_NAME}.app"
MACOS_DIR="${APP_BUNDLE}/Contents/MacOS"

rm -rf "${APP_BUNDLE}"
mkdir -p "${MACOS_DIR}"

cp Info.plist "${APP_BUNDLE}/Contents/Info.plist"

swiftc \
    -O \
    -framework AppKit \
    -framework GameController \
    -o "${MACOS_DIR}/${APP_NAME}" \
    main.swift

# Ad-hoc sign so macOS lets the bundle launch without quarantine prompts.
codesign --force --sign - "${APP_BUNDLE}"

echo "Built ${APP_BUNDLE}"
