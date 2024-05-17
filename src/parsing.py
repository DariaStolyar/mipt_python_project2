import os

ALPHA = ' 1234567890-йцукенгшщзхъфывапролджэёячсмитьбюqwertyuiopasdfghjklzxcvbnm'

with open(os.path.join('assets', 'dialogues.txt')) as f:
    content = f.read()


def clean_str(r: str) -> str:
    r = r[2:].lower()
    r = [c for c in r if c in ALPHA]
    return ''.join(r)


def get_mega_dataset() -> dict[str: list[tuple[str, str]]]:
    dataset = []  # [['Q', 'A'], ...]
    blocks = content.split('\n\n')
    for block in blocks:
        replicas = block.split('\n')[:2]
        if len(replicas) == 2:
            pair = (clean_str(replicas[0]), clean_str(replicas[1]))
            if pair[0] and pair[1]:
                dataset.append(pair)

    dataset = list(set(dataset))
    mega_dataset = {}  # word: [(Q, A), ...]

    for question, answer in dataset:
        words = question.split(' ')
        for word in words:
            if word not in mega_dataset:
                mega_dataset[word] = []
            mega_dataset[word].append((question, answer))
    return mega_dataset
