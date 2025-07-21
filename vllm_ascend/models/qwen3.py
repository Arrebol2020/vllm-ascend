from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm_ascend.quantization.quant_config import AscendLinearMethod

class CustomQwen3ForCausalLM(Qwen3ForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj", 
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    