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

    dialog = []  # Initialize an empty dialog list

    print("You can start chatting now. Type 'exit' to end the conversation.")

    while True:
        user_input = input("User: ")  # Take user input
        if user_input.strip().lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        # Add the user's message to the dialog
        dialog.append({"role": "user", "content": user_input})

        # Call the model to generate a response
        result = generator.chat_completion(
            [dialog],  # Pass the dialog as a list of one conversation
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[
            0
        ]  # Take the first (and only) result

        # Print the response
        print(f"Assistant: {result['generation']['content']}")

        # Add the assistant's message to the dialog
        dialog.append(result["generation"])

        if len(dialog) > 10:
            dialog = dialog[-10:]


if __name__ == "__main__":
    fire.Fire(interactive_chat)
