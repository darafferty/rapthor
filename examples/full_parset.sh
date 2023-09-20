#!/bin/sh -eu

# Grep for keys, optional (commented) keys, and sections;
# uncomment and convert ini to json
egrep '^((#[[:blank:]]*)?[[:alnum:]_]+ =|\[[^]]*\])' rapthor.parset \
  | sed 's,^# *,,' \
  > rapthor_full.parset
