# uniaccurate — Python binding

```bash
nimble clib                                              # build libUniAccurate.so
cd py && python3 setup.py build_ext --inplace            # build extension
cd py && python3 -m pytest -q                            # test
```

```python
import uniaccurate
uniaccurate.version()       # "0.1.0"
uniaccurate.fibonacci(10)   # 55
```
