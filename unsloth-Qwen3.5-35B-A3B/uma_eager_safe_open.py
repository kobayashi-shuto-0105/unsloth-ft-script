"""Optional DGX Spark / GB10 UMA safetensors load-peak workaround.

This is intentionally a small monkey patch around ``transformers.modeling_utils.safe_open``.
It is useful on unified-memory machines where safetensors mmap/page-cache pages and
CUDA tensors compete for the same physical memory pool during model loading.

Use from the training script with:

    from uma_eager_safe_open import install_uma_eager_safe_open
    install_uma_eager_safe_open()

The wrapper supports the safe_open methods currently used by Transformers 5.5:
``keys()``, ``get_tensor()``, ``get_slice()``, and ``metadata()``.
"""

from __future__ import annotations


def install_uma_eager_safe_open() -> None:
    """Install a Transformers safetensors ``safe_open`` wrapper.

    The wrapper eagerly materializes each model shard onto ``cuda:0`` and then
    hints Linux to drop the clean page cache for that shard using
    ``posix_fadvise(..., POSIX_FADV_DONTNEED)``.

    This is an experimental workaround. Disable it on non-UMA systems unless
    you specifically need to reduce load-time peak memory.
    """

    import gc
    import os
    from pathlib import Path

    import torch
    import transformers.modeling_utils as modeling_utils
    from safetensors import safe_open as original_safe_open

    current = getattr(modeling_utils, "safe_open", None)
    if getattr(current, "_uma_eager_safe_open", False):
        print("UMA eager safe_open already installed", flush=True)
        return

    def should_eager(filename: str) -> bool:
        name = Path(str(filename)).name
        return name.startswith("model.safetensors")

    def drop_page_cache(filename: str) -> None:
        try:
            fd = os.open(str(filename), os.O_RDONLY)
            try:
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    print(f"UMA_EAGER_SAFE_OPEN: posix_fadvise DONTNEED: {filename}", flush=True)
            finally:
                os.close(fd)
        except Exception as exc:  # pragma: no cover - best effort diagnostic
            print(f"WARNING: posix_fadvise failed for {filename}: {exc}", flush=True)

    class _EagerSafeOpen:
        _uma_eager_safe_open = True

        def __init__(self, filename, framework="pt", device="cpu"):
            self.filename = str(filename)
            self.framework = framework
            self.device = device
            self._delegate_cm = None
            self._delegate = None
            self._loaded = False
            self._keys: list[str] = []
            self._metadata = None
            self._tensors = {}
            self._eager = should_eager(self.filename)

            # Preserve normal behavior for non-model safetensors.
            if not self._eager:
                self._delegate_cm = original_safe_open(
                    self.filename,
                    framework=self.framework,
                    device=self.device,
                )
                self._delegate = self._delegate_cm.__enter__()

        def _ensure_loaded(self) -> None:
            if self._delegate is not None or self._loaded:
                return

            target = "cuda:0" if torch.cuda.is_available() else self.device
            print(f"UMA_EAGER_SAFE_OPEN: loading shard direct-to-{target}: {self.filename}", flush=True)

            with original_safe_open(self.filename, framework=self.framework, device=target) as handle:
                self._keys = list(handle.keys())
                try:
                    self._metadata = handle.metadata()
                except Exception:
                    self._metadata = None

                for key in self._keys:
                    tensor = handle.get_tensor(key)
                    if torch.cuda.is_available():
                        dev = getattr(tensor, "device", None)
                        if dev is not None and dev.type != "cuda":
                            tensor = tensor.to("cuda:0", non_blocking=True)
                    self._tensors[key] = tensor

            self._loaded = True
            drop_page_cache(self.filename)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def __enter__(self):
            if self._delegate is not None:
                return self._delegate
            self._ensure_loaded()
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._delegate_cm is not None:
                return self._delegate_cm.__exit__(exc_type, exc, tb)
            self._tensors.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

        def __del__(self):
            try:
                if self._delegate_cm is not None:
                    self._delegate_cm.__exit__(None, None, None)
            except Exception:
                pass
            try:
                self._tensors.clear()
            except Exception:
                pass

        def keys(self):
            if self._delegate is not None:
                return list(self._delegate.keys())
            self._ensure_loaded()
            return list(self._keys)

        def get_tensor(self, name):
            if self._delegate is not None:
                return self._delegate.get_tensor(name)
            self._ensure_loaded()
            return self._tensors[name]

        def get_slice(self, name):
            if self._delegate is not None:
                return self._delegate.get_slice(name)
            self._ensure_loaded()
            return self._tensors[name]

        def metadata(self):
            if self._delegate is not None:
                try:
                    return self._delegate.metadata()
                except Exception:
                    return None
            self._ensure_loaded()
            return self._metadata

    modeling_utils.safe_open = _EagerSafeOpen
    print("Installed UMA eager safe_open patch for transformers.modeling_utils.safe_open", flush=True)
