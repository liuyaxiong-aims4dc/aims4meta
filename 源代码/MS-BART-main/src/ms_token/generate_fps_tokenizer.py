from transformers import AutoTokenizer


if __name__ == "__main__":

    selfies_tokenizer = AutoTokenizer.from_pretrained("./logs/tokenizer/zju-selfies-tokenizer")
    selfies_tokenizer.add_tokens(["<fps_sep>"])
    fps_tokens = []
    for idx in range(4096):
        fps_tokens.append(f"<fp{idx:04d}>")
    selfies_tokenizer.add_tokens(fps_tokens)
    selfies_tokenizer.save_pretrained(f"./logs/tokenizer/zju-selfies-fps-tokenizer")
