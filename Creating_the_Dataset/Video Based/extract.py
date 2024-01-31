import numpy as np

"""min_loss=np.Inf
idx=0
with open('hps.txt') as fo:
    for i,rec in enumerate(fo):
        if rec[:12]=='Minimum Loss':
            if min_loss>float(rec[14:]):
                min_loss=float(rec[14:])
                idx = i
print(i,min_loss)"""

import numpy as np

min_loss = np.Inf
idx = 0
with open('C:\\Users\\David\\Desktop\\outl.txt', 'w') as o:
    with open('C:\\Users\\David\\Desktop\\l.txt') as f:
        for line in f:
            line_out = ""
            for word in line.split("  "):
                if word != "":
                    if word[0]=="C" or word[:2]==" C":
                        word="Conv2d"
                    if word[0]=="[" or word[:2]==" [C]":
                        word="["+word[4:]
                    line_out = line_out + " &" + word
            line_out = line_out[2:] + r'\\\hline '
            if line_out[1] != 'R' and line_out[1] != 'B' and line_out[0] != 'R' and line_out[0] != 'B':
                print(line_out[1],line_out[2])
                print(line_out)
                o.write(line_out)
