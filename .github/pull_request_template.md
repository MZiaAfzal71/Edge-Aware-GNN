## Purpose

Describe the research or repository-contract change.

## Validation

- [ ] `edgegnn validate --root .`
- [ ] `edgegnn publication --tables-only --output /tmp/edgegnn-publication`
- [ ] `pytest -q`
- [ ] No notebooks, `*.egg-info`, large derived prediction tables, or publication outputs added
- [ ] Benchmark-result changes include matching metrics, predictions, metadata, and provenance
