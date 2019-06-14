import pandas as pd
# Match ID with their respective sentences

sentenes = pd.read_csv("../data/actual.csv")
excel = pd.read_csv("../data/sentences.csv")

ids = []
sens = []
for index in range(0, len(excel)):
    id = int(excel.at[index, "ID"])
    if id <= 262:
        sen = str(sentenes.at[id - 1, "sentence"])
    else:
        sen = str(sentenes.at[id - 2, "sentence"])

    ids.append(id)
    sens.append(sen)

dictionary = {
    "ID": ids,
    "Sentence": sens
}

df = pd.DataFrame(data=dictionary)
df.to_csv("actual2.csv")


