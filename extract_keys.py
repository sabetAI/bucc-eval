import os
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('dir')

args = argparser.parse_args()
files = os.listdir(args.dir)
lang1, lang2 = args.dir.split('-')

idmap = {}

for fname in files:
    if 'gold' in fname:
        with open(os.path.join(args.dir, fname), encoding='utf-8') as fp:
            for line in fp:
                lang1_id, lang2_id = line.split('\t')
                lang1_id = int(lang1_id.split('-')[1].strip())
                lang2_id = int(lang2_id.split('-')[1].strip())
                idmap[lang1_id-1] = lang2_id-1


with open(os.path.join(args.dir, 'bucc2018.' + args.dir + '.gold'), 'w') as fp:
    for l1key, l2key in idmap.items():
        fp.write(str(l1key) + '\t' + str(l2key) + '\n')
