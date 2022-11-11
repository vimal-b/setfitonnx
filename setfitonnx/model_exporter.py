
from loguru import logger
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from joblib import load
import os
import sclblonnx as so
from .config import model_config
from .onnxexporter import ONNXExporter
from .pooler import MeanPoolingOnnx


def convert_onnx(model_path:str,output_dir:str,quantize:bool=False,opset_version=12):
    
    
    # Check whether the specified output directory exists or not
    isExist = os.path.exists(output_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(output_dir)
    onnx_exporter = ONNXExporter()
    logger.info("Exporting to ONNX...")
    pooled_model_pytorch = MeanPoolingOnnx(model_path)
    # logger.info("Mean Pooling Model is built successfully!")
    # logger.info("Exporting Pooled Model to ONNX")
    pooled_model_onnx = onnx_exporter(model=pooled_model_pytorch,tokenizer_path=model_path,output=output_dir+"/embeddings.onnx",quantize=quantize,opset=opset_version)
    # logger.info("ONNX Model Successfully built in the directory '{}'".format(output_dir))
    # logger.info("Exporting model_head.pkl to ONNX...")
    # Get the input shape of the model
    model_info = model_config(model_path)
    output_shape = model_info.get('hidden_size',None)
    if not output_shape:
        raise Exception("Hidden Size detail not available in model config.json")
    initial_type = [('float_input', FloatTensorType([output_shape]))]
    try:
        model_head = load(model_path+'/model_head.pkl')
    except:
        logger.error("model_head.pkl file is not found in the model path directory")

    model_head_onnx = convert_sklearn(model_head, initial_types=initial_type)

    with open(output_dir+"/model_head.onnx", "wb") as f:
        f.write(model_head_onnx.SerializeToString())
    # logger.info("Model Head Exported to ONNX Successfully")

    # logger.info("Merging both Onnx Models...")
    # Read both the onnx files
    model_graph = so.graph_from_file(output_dir+"/embeddings.onnx") 
    head_graph = so.graph_from_file(output_dir+"/model_head.onnx")
    merged_model = so.merge(sg1=model_graph,sg2=head_graph,io_match=[("embeddings","float_input")],complete=False,_verbose=False)
    so.graph_to_file(merged_model,output_dir+'/setfit-model.onnx')
    # Deleting individual models
    try:
        os.remove(output_dir+"/embeddings.onnx")
        os.remove(output_dir+"/model_head.onnx")
    except:
        pass
    logger.info("ONNX Model Successfully Built in Directory:{}".format(output_dir))


