import pandas as pd


if __name__ == "__main__":
    with open("TinyStoriesV2-GPT4-train.txt", "r") as f:
        data = f.read()
    data = data.replace("\n", "")
    data = data.split("<|endoftext|>")
    data = data[:500_000]
    df = pd.DataFrame(data, columns=["text"])
    print()
    df.to_parquet("tinystories_train.parquet")