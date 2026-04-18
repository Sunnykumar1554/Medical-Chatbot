import pathlib
text = pathlib.Path(r'c:\Users\sunny\Desktop\chatbot\app.py').read_text(encoding='utf-8')
print(text.splitlines()[-5:])
for i,line in enumerate(text.splitlines(), start=1):
    if i>=65:
        print(i,repr(line))
