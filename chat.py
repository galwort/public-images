from typing import Optional
import fire
from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 8192,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialog = []

    while True:
        user_dialog = {"role": "user", "content": input("User: ")}
        dialog.append(user_dialog)

        results = generator.chat_completion(
            [dialog],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        response = results[0]["generation"]["content"]
        print("\nLlama:", response)
        print("\n==================================\n")

        dialog.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    fire.Fire(main)
