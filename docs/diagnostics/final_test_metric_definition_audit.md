# Final-test metric-definition audit

The frozen protocol requires target-aware denominators:

- tool-call JSON parse: all target rows requiring tool calls, including mixed
  targets; wrong output kind is failure
- response-only wrapper: response-only target rows only
- auxiliary all-row counts may be reported separately

The historical canonical metric files use the field names tool_parse and
response_wrapper for auxiliary all-row counts. The CPU raw re-audit matches
their numerical values without editing those historical files. The new
scripts/analyze_final_test.py emits both:

1. protocol-primary target-aware metrics; and
2. explicitly labelled auxiliary all-row counts.

This prevents a denominator change from being hidden inside the final test.
