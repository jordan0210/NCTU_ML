def load_label(filename):
    file = open(filename, 'r')
    lines = file.read()
    label = []
    for line in lines.split('\n'):
        if line.strip():
            label.append(int(line))
    return label

def load_data(filename):
    file = open(filename, 'r')
    lines = file.read()
    data = []
    for line in lines.split('\n'):
        pixels = line.split(',')
        data.append({(i+1):float(pixels[i]) for i in range(len(pixels)) if pixels[i].strip()})
    data.pop()
    return data