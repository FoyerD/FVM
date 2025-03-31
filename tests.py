from itertools import product
sett = set(map(lambda x: "".join(x), set(product('1234', repeat=2))))
print(*sett)