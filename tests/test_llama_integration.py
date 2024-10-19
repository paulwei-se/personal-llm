import asyncio
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.llama_model import LlamaModel
from src.document_intelligence import DocumentIntelligence

async def test_llama_basic():
    llm = LlamaModel()
    await llm.warmup()  # Add warmup phase

    prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks briefly.",
        "What is deep learning?"
    ]
    
    for prompt in prompts:
        start_time = time.time()
        response = await llm.generate(prompt)
        end_time = time.time()
        
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print(f"Current GPU stats: {llm.gpu_monitor.get_stats()}")
        print("-" * 80)

async def main():
    print("Starting Llama Integration Tests\n")
    
    try:
        
        await test_llama_basic()
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    finally:
        print("Tests completed")

if __name__ == "__main__":
    asyncio.run(main())