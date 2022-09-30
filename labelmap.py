label_map = {'good': 0, 'bad': 1, 'timeout': 2, 'horns': 3, 'victory': 4}
reversed_map = {v: k for k, v in label_map.items()}
print(reversed_map)