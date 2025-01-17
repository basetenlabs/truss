# Truss-Transfer

Python-optional download utility


```base
pip install truss_transfer
# pip install /workspace/model-performance/michaelfeil/truss/truss-transfer/target/wheels/truss_transfer-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl
```

```python
import truss_transfer

def lazy_data_loader(download_dir: str, num_workers: int = 4):
    print(f"download using {truss_transfer.__version__}")
    try:
        truss_transfer.lazy_data_resolve(str(download_dir), int(num_workers))
    except Exception as e:
        print(f"Lazy data resolution failed: {e}")
        raise
```

### Installing the CLI as binary

```
cargo build --release --bin truss_transfer_cli --features cli
```

### Running the binary
```
./target/release/truss_transfer_cli /tmp/ptr 4
```

### Building a wheel from source

Prerequisites:
```sh
# apt-get install patchelf
# Install rust via Rustup https://www.rust-lang.org/tools/install
pip install maturin==1.7.8
```

This will build you the wheels for your current `python3 --version`.
The output should look like this:
```
maturin build --release
🔗 Found pyo3 bindings
🐍 Found CPython 3.9 at /workspace/model-performance/michaelfeil/.asdf/installs/python/3.9.21/bin/python3
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.75s
🖨  Copied external shared libraries to package truss_transfer.libs directory:
    /usr/lib/x86_64-linux-gnu/libssl.so.3
    /usr/lib/x86_64-linux-gnu/libcrypto.so.3
📦 Built wheel for CPython 3.9 to /workspace/model-performance/michaelfeil/truss/truss-transfer/target/wheels/truss_transfer-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl
```
