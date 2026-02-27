from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


class TrtConfig(object):
    def __init__(self, engine_path, input_size=320):
        self.engine_path = Path(engine_path)
        self.input_size = input_size


class TrtInferencer(object):
    """TensorRT inference wrapper."""

    def __init__(self, config):
        """Create inferencer with lazy-loaded TensorRT/CUDA context."""
        self.config = config
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.input_binding = None
        self.output_bindings = []

        self._trt = None
        self._cuda = None
        self._pycuda_autoinit = None

    def load(self):
        """Load TensorRT engine and allocate device/host buffers."""
        if not self.config.engine_path.exists():
            raise FileNotFoundError(
                "TensorRT engine not found: {}".format(self.config.engine_path)
            )

        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT runtime dependencies are missing. "
                "Install/launch in Jetson TensorRT environment."
            ) from exc

        self._trt = trt
        self._cuda = cuda
        self._pycuda_autoinit = pycuda.autoinit

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.config.engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine: {}".format(
                    self.config.engine_path
                )
            )

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        stream = cuda.Stream()
        bindings = [0] * engine.num_bindings
        inputs = []
        outputs = []

        for idx in range(engine.num_bindings):
            dtype = trt.nptype(engine.get_binding_dtype(idx))
            shape = list(context.get_binding_shape(idx))

            if any(dim < 0 for dim in shape):
                profile_shape = engine.get_profile_shape(0, idx)
                shape = list(profile_shape[2])
                context.set_binding_shape(idx, tuple(shape))

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[idx] = int(dev_mem)

            binding = {
                "index": idx,
                "shape": tuple(shape),
                "host": host_mem,
                "device": dev_mem,
            }

            if engine.binding_is_input(idx):
                inputs.append(binding)
            else:
                outputs.append(binding)

        if len(inputs) != 1:
            raise RuntimeError("Expected one input binding, got {}".format(len(inputs)))
        if not outputs:
            raise RuntimeError("No output bindings found in TensorRT engine")

        self.engine = engine
        self.context = context
        self.stream = stream
        self.bindings = bindings
        self.input_binding = inputs[0]
        self.output_bindings = outputs

    @property
    def input_shape(self):
        """Return NCHW input shape resolved from TensorRT bindings."""
        if self.input_binding is None:
            raise RuntimeError("TrtInferencer is not loaded")
        return tuple(int(x) for x in self.input_binding["shape"])

    def _preprocess(self, frame):
        """Resize + normalize BGR frame to NCHW float32 tensor."""
        if self.input_binding is None:
            raise RuntimeError("TrtInferencer is not loaded")

        _, _, in_h, in_w = self.input_shape
        resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return np.expand_dims(chw, axis=0)

    def infer(self, frame):
        """Run inference for one frame and return first output tensor and latency in ms."""
        if self.input_binding is None or self.context is None or self.stream is None:
            raise RuntimeError("TrtInferencer is not loaded")
        if self._cuda is None:
            raise RuntimeError("CUDA runtime is not available")

        blob = self._preprocess(frame)
        np.copyto(self.input_binding["host"], blob.ravel())

        t0 = perf_counter()
        self._cuda.memcpy_htod_async(
            self.input_binding["device"], self.input_binding["host"], self.stream
        )
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        for out in self.output_bindings:
            self._cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        infer_ms = (perf_counter() - t0) * 1000.0

        first_out = self.output_bindings[0]
        output = np.array(first_out["host"], copy=False).reshape(first_out["shape"])
        return output, infer_ms
