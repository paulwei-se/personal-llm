import torch
import torch.amp
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig
from typing import List, Dict, Any
import logging
import asyncio
from src.utils.gpu_monitor import GPUMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModel:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", device: str = None):
        self.model_name = model_name
        self.gpu_monitor = GPUMonitor()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Enable better memory efficiency
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
            
        logger.info(f"Initializing Llama model on {self.device}")
        
        # Start GPU monitoring
        self.gpu_monitor.start()
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
                
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add generation config
        self.generation_config = {
            "max_new_tokens": 256,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
            "use_cache": True,
            "num_return_sequences": 1,
        }

        config = LlamaConfig.from_pretrained(model_name)
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory={0: "7.0GB"},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # Optimize model placement
        self.model = self.model.to(self.device)
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f"Model device map: {self.model.hf_device_map}")

        
        # Log initial memory usage
        logger.info(f"Initial GPU stats: {self.gpu_monitor.get_stats()}")
        
    def __del__(self):
        self.gpu_monitor.stop()

    async def generate(
        self, 
        prompt: str, 
        max_length: int = None,
        **kwargs
    ) -> str:
        """Generate text using the Llama model"""
        if max_length is None:
            max_length = self.generation_config["max_new_tokens"]
            
        pre_stats = self.gpu_monitor.get_stats()
        logger.info(f"Pre-generation GPU stats: {pre_stats}")
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        logger.debug(f"LlamaModel received prompt: {prompt}")
        
        response = await asyncio.to_thread(
            self._generate_sync,
            inputs,
            max_length,
            **kwargs
        )
        logger.debug(f"LlamaModel generated response: {response}")

        post_stats = self.gpu_monitor.get_stats()
        logger.info(f"Post-generation GPU stats: {post_stats}")
        
        return response

    def _generate_sync(
        self, 
        inputs: Dict[str, torch.Tensor],
        max_length: int = None,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ) -> str:
        """Synchronous generation function with default values"""
        if max_length is None:
            max_length = self.generation_config["max_new_tokens"]
        if temperature is None:
            temperature = self.generation_config["temperature"]
        if top_p is None:
            top_p = self.generation_config["top_p"]
            
        generation_kwargs = {
            **self.generation_config,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
        
         # Get the length of the input prompt in tokens
        prompt_length = inputs['input_ids'].shape[1]
        
        # Decode only the newly generated tokens, don't repeat the prompt
        response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
                
        return response
    
    async def warmup(self):
        """Perform model warm-up to optimize GPU compilation"""
        warmup_prompt = "Hello, how are you?"
        for _ in range(3):
            await self.generate(warmup_prompt, max_length=32)