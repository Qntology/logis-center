import gguf
writer = gguf.GGUFWriter("test.gguf", "test")
print("GGUFWriter methods:")
for m in dir(writer):
    if not m.startswith("_"):
        print(f" - {m}")
