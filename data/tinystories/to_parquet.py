import pandas as pd


if __name__ == "__main__":
    with open("TinyStoriesV2-GPT4-train.txt", "r") as f:
        data = f.read()
    data = data.replace("\n", "")
    data = data.split("<|endoftext|>")
    data = data[:150_000]
    print(len(data))
    df = pd.DataFrame(data, columns=["text"])
    print(len(df))
    df.to_parquet("150k_tinystories_train.parquet")