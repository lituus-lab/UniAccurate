<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# uniaccurate — Python binding

```bash
nimble pyLib                                             # build the shared lib the extension links against (.so / .dylib / .dll per OS)
cd py && python3 setup.py build_ext --inplace            # build extension
cd py && python3 -m pytest -q                            # test
```

```python
import uniaccurate
uniaccurate.version()                  # "1.0.0"
uniaccurate.two_sum(1.0, 2e16)         # (2e16, 1.0) — error recovered
```
