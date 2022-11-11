from torch.onnx import export
from itertools import chain
from collections import OrderedDict
from torch import nn
from .pooler import MeanPoolingOnnx
from transformers import AutoTokenizer
from io import BytesIO

# Check for onnxruntime 
try:
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
    from onnxruntime.quantization import quantize_dynamic

    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False



class ONNXExporter:
    """
        Exports the Sentence Transformer Model to ONNX Graph.
    """

    def __call__(self, model:MeanPoolingOnnx, tokenizer_path:str,output=None, quantize=False, opset=12):
        """
        Exports the Pooled Sentence Transformer model to ONNX.

        Args:
            model: Sentence transformer nn.Module with mean pooling layer
            output: optional output model path, defaults to return byte array if None
            quantize: if model should be quantized (requires onnx to be installed), defaults to False
            opset: onnx opset, defaults to 12

        Returns:
            path to model output or model as bytes depending on output parameter
        """

        #Model Inputs
        inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )
        #Model Outputs
        outputs = OrderedDict({"embeddings": {0: "batch", 1: "sequence"}})

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Generate dummy inputs
        dummy = dict(tokenizer(["test inputs"], return_tensors="pt"))

        # Default to BytesIO if no output file provided
        output = output if output else BytesIO()

        # Export model to ONNX
        export(
            model,
            (dummy,),
            output,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(inputs.keys()),
            output_names=list(outputs.keys()),
            dynamic_axes=dict(chain(inputs.items(), outputs.items())),
        )

        # Quantize model
        if quantize:
            if not ONNX_RUNTIME:
                raise ImportError(
                    'onnxruntime is not available "pip install onnxruntime"')

            output = self.quantization(output)

        if isinstance(output, BytesIO):
            # Reset stream and return bytes
            output.seek(0)
            output = output.read()

        return output
