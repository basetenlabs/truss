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
PACKAGE_NAME="truss"  # Hardcoded for now

# Function to check if a specific version exists on PyPI
version_exists_on_pypi() {
  local version="$1"
  curl -s -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/${PACKAGE_NAME}/${version}/json" | grep -q "200"
}

# Function to increment version based on bump type
bump_version() {
  local bump_type="$1"
  local major="$2"
  local minor="$3"
  local patch="$4"

  case "$bump_type" in
    major)
      echo "$((major + 1)) 0 0"
      ;;
    minor)
      echo "$major $((minor + 1)) 0"
      ;;
    patch)
      echo "$major $minor $((patch + 1))"
      ;;
    *)
      echo "Invalid bump type: $bump_type. Use 'major', 'minor', or 'patch'."
      exit 1
      ;;
  esac
}

# Strip any suffix beyond the semver (retain only X.Y.Z)
semver=$(echo "$CURRENT_VERSION" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')

# Split the semver into major, minor, and patch
IFS='.' read -r major minor patch <<< "$semver"

# Bump the version based on the bump type argument
read -r major minor patch <<< "$(bump_version "$BUMP_TYPE" "$major" "$minor" "$patch")"

# Find next available version, accounting for yanked releases
while version_exists_on_pypi "${major}.${minor}.${patch}"; do
  echo "Version ${major}.${minor}.${patch} already exists on PyPI, trying next..."
  read -r major minor patch <<< "$(bump_version "$BUMP_TYPE" "$major" "$minor" "$patch")"
done

# Construct the new version string
new_version="${major}.${minor}.${patch}"
echo "Using version: $new_version"

# Set the new version using Poetry
poetry version "$new_version"

# If GITHUB_OUTPUT is set (GitHub Actions context), write the output there.
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "version=${new_version}" >> "${GITHUB_OUTPUT}"
fi
