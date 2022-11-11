# SetFitONNX
## Export the setfit model to ONNX format


## Features

- Mean Pooling Layer to ONNX Graph
- Single ONNX Model combining both the sentence transformer & classification head

## Installation
```sh
pip install setfitonnx
```
## How to 
```python
from setfitonnx import convert_onnx

# Setfit Model directory
model_path = "/home/mysetfit-model"
# ONNX Output directory
output_dir = "/home/setfit-onnx-model"
# Convert to ONNX
convert_onnx(model_path=model_path,output_dir=output_dir)

```

## License

MIT