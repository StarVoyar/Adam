SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")
rm -rf "$ROOT/src/data/"*
