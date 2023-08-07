from typing import Optional
import fire
from llama import Llama


def main(
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

    dialog = [
        {
            "role": "system",
            "content": "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature. "
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information. ",
        },
    ]

    while True:
        print("Start of loop:\n", dialog)
        user_dialog = {"role": "user", "content": input("User: ")}
        print("After user input:\n", dialog)
        dialog.append(user_dialog)
        print("After appending user input:\n", dialog)

        results = generator.chat_completion(
            [dialog],  # Wrapping dialog inside a list
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        print("After chat completion:\n", dialog)

        response = results[0]["generation"]["content"]  # Removing curly braces
        print("Llama: ", response)
        print("After response:\n", dialog)
        print("\n==================================\n")

        dialog.append({"role": "assistant", "content": response})
        print("After appending response:\n", dialog)


if __name__ == "__main__":
    fire.Fire(main)
