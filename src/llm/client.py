import os
import openai
from typing import List, Dict, Any, Optional, Tuple
import tiktoken

class LLMClient:
    """
    Unified LLM client supporting OpenAI (GPT-4o, GPT-4o-mini) and 
    compatible APIs (Qwen-2.5-7B/32B-Instruct via OpenAI-compatible endpoints).
    
    Tracks token usage for cost analysis.
    """
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, base_url: Optional[str] = None, verbose: bool = False):
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        
        # Token counting
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0  # Track number of LLM calls
        self.verbose = verbose  # Whether to print token usage per call
        
        # Initialize tokenizer for token estimation
        try:
            if "gpt-4" in model.lower():
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model.lower():
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # For Qwen or other models, use cl100k_base as approximation
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512, stop: Optional[List[str]] = None) -> str:
        """Generate completion and track token usage."""
        try:
            self.call_count += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
            
            # Track token usage for this call
            if hasattr(response, 'usage') and response.usage:
                current_input_tokens = response.usage.prompt_tokens
                current_output_tokens = response.usage.completion_tokens
            else:
                # Fallback: estimate tokens
                current_input_tokens = self._count_tokens(messages)
                output_content = response.choices[0].message.content
                current_output_tokens = self._count_tokens([{"role": "assistant", "content": output_content}])
            
            current_total = current_input_tokens + current_output_tokens
            
            # Update cumulative totals
            self.total_input_tokens += current_input_tokens
            self.total_output_tokens += current_output_tokens
            cumulative_total = self.total_input_tokens + self.total_output_tokens
            
            # Print token usage if verbose mode is enabled
            if self.verbose:
                print(f"\n{'─'*60}")
                print(f"[LLM Call #{self.call_count}] Token Usage:")
                print(f"  This call - Input: {current_input_tokens:,} | Output: {current_output_tokens:,} | Total: {current_total:,}")
                print(f"  Cumulative - Input: {self.total_input_tokens:,} | Output: {self.total_output_tokens:,} | Total: {cumulative_total:,}")
                print(f"{'─'*60}\n")
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""
    
    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count for messages."""
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
        num_tokens += 2  # Every reply is primed with <im_start>assistant
        return num_tokens
    
    def get_token_usage(self) -> Dict[str, int]:
        """Return total token usage statistics."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
    
    def reset_token_count(self):
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
