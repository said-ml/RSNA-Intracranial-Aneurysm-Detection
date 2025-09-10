import pandas as pd
LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]
df = pd.read_csv(r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation/labels.csv")
print(f'labels shape ={df.shape}')
for colmn in LABEL_COLS:
    print(f' positive case of {colmn} is {(df[colmn]==1).sum()}')
    print(f' negative case of {colmn} is {(df[colmn]==0).sum()}')

import matplotlib.pyplot as plt

positives = {
    "L-ICA (Infraclinoid)": 74,
    "R-ICA (Infraclinoid)": 90,
    "L-ICA (Supraclinoid)": 293,
    "R-ICA (Supraclinoid)": 250,
    "L-MCA": 203,
    "R-MCA": 271,
    "AComm": 344,
    "L-ACA": 46,
    "R-ACA": 54,
    "L-PComm": 84,
    "R-PComm": 100,
    "Basilar Tip": 110,
    "Post. Circulation": 110,
    "Any Aneurysm": 1722
}

plt.figure(figsize=(10,6))
plt.bar(positives.keys(), positives.values())
plt.xticks(rotation=90)
plt.ylabel("Positive Cases")
plt.title("Aneurysm Labels Distribution")
plt.show()
