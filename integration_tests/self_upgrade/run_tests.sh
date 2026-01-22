#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SKIP_ACTUAL_UPGRADE=${SKIP_ACTUAL_UPGRADE:-0}
ENVIRONMENTS=${ENVIRONMENTS:-"uv_tool uv_venv pipx conda"}
TRUSS_VERSION=${TRUSS_VERSION:-"0.12.8rc500"}

log() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    log "Cleaning up containers and images..."
    docker rm -f truss-upgrade-test 2>/dev/null || true
    for env in $ENVIRONMENTS; do
        docker rmi -f "truss-upgrade-test-$env" 2>/dev/null || true
    done
}

trap cleanup EXIT

run_env_test() {
    local env_name=$1
    local dockerfile=$2
    local build_args=$3

    log "=============================================="
    log "Testing environment: $env_name"
    log "=============================================="

    log "Building image..."
    docker build -f "$dockerfile" $build_args --build-arg TRUSS_VERSION="$TRUSS_VERSION" -t "truss-upgrade-test-$env_name" . || {
        error "Failed to build image for $env_name"
        return 1
    }

    log "Running tests..."
    docker run --rm \
        -e SKIP_ACTUAL_UPGRADE="$SKIP_ACTUAL_UPGRADE" \
        -e TRUSS_IGNORE_PRERELEASE_CHECK=1 \
        --name truss-upgrade-test \
        "truss-upgrade-test-$env_name" || {
        error "Tests failed for $env_name"
        return 1
    }

    log "✅ $env_name tests passed"
    return 0
}

main() {
    echo "=============================================="
    echo "Truss Self-Upgrade Integration Tests"
    echo "=============================================="
    echo "Settings:"
    echo "  SKIP_ACTUAL_UPGRADE=$SKIP_ACTUAL_UPGRADE"
    echo "  ENVIRONMENTS=$ENVIRONMENTS"
    echo "  TRUSS_VERSION=$TRUSS_VERSION"
    echo ""

    local failed=0
    local passed=0

    for env in $ENVIRONMENTS; do
        case $env in
            uv_tool)
                if run_env_test "uv_tool" "Dockerfile.uv" "--build-arg INSTALL_METHOD=tool"; then
                    ((passed+=1))
                else
                    ((failed+=1))
                fi
                ;;
            uv_venv)
                if run_env_test "uv_venv" "Dockerfile.uv" "--build-arg INSTALL_METHOD=venv"; then
                    ((passed+=1))
                else
                    ((failed+=1))
                fi
                ;;
            pipx)
                if run_env_test "pipx" "Dockerfile.pipx" ""; then
                    ((passed+=1))
                else
                    ((failed+=1))
                fi
                ;;
            conda)
                if run_env_test "conda" "Dockerfile.conda" ""; then
                    ((passed+=1))
                else
                    ((failed+=1))
                fi
                ;;
            *)
                warn "Unknown environment: $env"
                ;;
        esac
    done

    # Run settings TOML tests (environment-agnostic, only need to run once)
    if [[ " $ENVIRONMENTS " == *" uv_venv "* ]]; then
        log "=============================================="
        log "Running settings TOML tests (using uv_venv image)"
        log "=============================================="
        if docker run --rm -e TRUSS_IGNORE_PRERELEASE_CHECK=1 "truss-upgrade-test-uv_venv" python /tests/test_settings_toml.py; then
            log "✅ settings_toml tests passed"
            ((passed+=1))
        else
            error "settings_toml tests failed"
            ((failed+=1))
        fi
    fi

    echo ""
    echo "=============================================="
    echo "Summary"
    echo "=============================================="
    echo -e "Passed: ${GREEN}$passed${NC}"
    echo -e "Failed: ${RED}$failed${NC}"

    if [ $failed -gt 0 ]; then
        exit 1
    fi
}

main "$@"
