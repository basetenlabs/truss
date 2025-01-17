# Truss-Transfer

Python-optional download utility


```base
pip install truss_transfer
# pip install /workspace/model-performance/michaelfeil/truss/truss-transfer/target/wheels/truss_transfer-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl
```

```python
import truss_transfer
print(f"downloading using version: {truss_transfer.__version__}")
truss_transfer.lazy_data_resolve(download_dir=str("/tmp"), num_workers=int(4))
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
python3 --version
# apt-get install patchelf
# Install rust via Rustup https://www.rust-lang.org/tools/install
pip install maturin
```

This will build you the wheels for your current `python3 --version`
```
maturin build --relase
🔗 Found pyo3 bindings
🐍 Found CPython 3.9 at /workspace/model-performance/michaelfeil/.asdf/installs/python/3.9.21/bin/python3
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.75s
🖨  Copied external shared libraries to package truss_transfer.libs directory:
    /usr/lib/x86_64-linux-gnu/libssl.so.3
    /usr/lib/x86_64-linux-gnu/libcrypto.so.3
📦 Built wheel for CPython 3.9 to /workspace/model-performance/michaelfeil/truss/truss-transfer/target/wheels/truss_transfer-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl
```
