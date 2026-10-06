#!/bin/bash
# Print dependency versions recorded in a Rapthor Docker image's labels.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 IMAGE" >&2
  exit 1
fi

docker image inspect --format '{{range $key, $value := .Config.Labels}}{{printf "%s=%s\n" $key $value}}{{end}}' "$1" |
  LC_ALL=C awk -F= '
    /^nl\.astron\.rapthor\.[[:alnum:]-]+\.version=/ {
      name = $1
      sub(/^nl\.astron\.rapthor\./, "", name)
      sub(/\.version$/, "", name)
      gsub(/-/, "", name)
      print toupper(name) "_COMMIT=" substr($0, index($0, "=") + 1)
      found = 1
    }
    END {
      if (!found) {
        print "Error: no Rapthor version labels found in image." > "/dev/stderr"
        exit 1
      }
    }
  '
