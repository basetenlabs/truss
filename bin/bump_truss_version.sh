#!/bin/bash
# This script bumps the project version using Poetry.
# It extracts the current version from `poetry version`, strips any non-semver suffix,
# and then increments the major, minor, or patch version as specified.
#
# Usage:
#   ./bump_version.sh         # bumps the patch (micro) version by default
#   ./bump_version.sh major   # bumps the major version (resets minor and patch to 0)
#   ./bump_version.sh minor   # bumps the minor version (resets patch to 0)

set -euo pipefail

# Default bump type is "patch" (micro)
BUMP_TYPE="${1:-patch}"

# Get the current version using Poetry.
# Expected output format: "truss 0.9.60rc006"
version_output=$(poetry version)

# Extract the version (second field)
current_version=$(echo "$version_output" | awk '{print $2}')

# Strip any suffix beyond the semver (retain only X.Y.Z)
semver=$(echo "$current_version" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')

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
