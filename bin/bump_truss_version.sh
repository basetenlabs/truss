#!/bin/bash
# This script bumps the project version based on an input version number.
# It takes an existing version number as an argument instead of extracting it via Poetry
# and increments the major, minor, or patch version as specified.
#
# Usage:
#   ./bump_version.sh 0.9.60         # bumps the patch (micro) version by default -> 0.9.61
#   ./bump_version.sh 0.9.60 major   # bumps the major version (resets minor and patch to 0) -> 1.0.0
#   ./bump_version.sh 0.9.60 minor   # bumps the minor version (resets patch to 0) -> 0.10.0

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <current_version> [major|minor|patch]"
  exit 1
fi

CURRENT_VERSION="$1"
BUMP_TYPE="${2:-patch}"  # Default bump type is "patch" (micro)

# Strip any suffix beyond the semver (retain only X.Y.Z)
semver=$(echo "$CURRENT_VERSION" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')

# Split the semver into major, minor, and patch
IFS='.' read -r major minor patch <<< "$semver"

# Bump the version based on the bump type argument
case "$BUMP_TYPE" in
  major)
    major=$((major + 1))
    minor=0
    patch=0
    ;;
  minor)
    minor=$((minor + 1))
    patch=0
    ;;
  patch)
    patch=$((patch + 1))
    ;;
  *)
    echo "Invalid bump type: $BUMP_TYPE. Use 'major', 'minor', or 'patch'."
    exit 1
    ;;
esac

# Construct the new version string
new_version="${major}.${minor}.${patch}"

# Set the new version using Poetry
poetry version "$new_version"

# If GITHUB_OUTPUT is set (GitHub Actions context), write the output there.
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "version=${new_version}" >> "${GITHUB_OUTPUT}"
fi
