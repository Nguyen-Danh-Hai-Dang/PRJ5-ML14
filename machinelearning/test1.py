import numpy as np

data = np.load("machineleadata/lang_id.npz")

languages = data["language_names"]

for split in ["train", "dev", "test"]:

    y = data[f"{split}_y"]

    print(f"\n{split.upper()} DISTRIBUTION")

    for i, language in enumerate(languages):

        count = np.sum(y == i)

        print(language, ":", count)