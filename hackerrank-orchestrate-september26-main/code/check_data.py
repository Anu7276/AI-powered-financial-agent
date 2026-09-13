import pandas as pd

img = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\images.csv')
msg = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\messages.csv')
print("Images count:", len(img))
print(img.head(10))
print("\nMessages count:", len(msg))
print(msg.head(10))
