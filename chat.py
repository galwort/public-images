from typing import Optional
import fire
from llama import Llama


def interactive_chat(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print("You can start chatting now. Type 'exit' to end the conversation.")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        print(f"Processing: {user_input}")  # Debugging line

        # Try generating a fixed response first
        print("Assistant: Hello!")

        # Optionally, uncomment the following lines to enable model response
        # dialog = [{"role": "user", "content": user_input}]
        # result = generator.chat_completion([dialog], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)[0]
        # print(f"Assistant: {result['generation']['content']}")


if __name__ == "__main__":
    fire.Fire(interactive_chat)
