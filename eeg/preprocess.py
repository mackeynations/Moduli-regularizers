# conda env: mnisteeg
import numpy as np
import csv

mydict = {}
mydictstd = {}


with open('data/EP1.01.txt') as myfile:
    k = 0 
    for i in myfile:
        line = i.split('\t')
        #print(line[3])
        k += 1
        if k < 15:
            mydict[line[3]] = []
            mydictstd[line[3]] = []
        if k == 15:
            break



with open('data/EP1.01.txt') as myfile:
    with open('data/tocsv.csv', 'a') as writefile: 
        with open('data/sols.csv', 'a') as solfile:
            writer = csv.writer(writefile)
            writer2 = csv.writer(solfile)
            for i in myfile:
                line = i.split('\t')
                newt = line[-1].split(',')[:130]
                writer.writerow(newt)
                sols = [int(line[4])+1]
                writer2.writerow(sols)
                mydict[line[3]].append(np.mean([float(m) for m in newt]))
                mydictstd[line[3]].append(np.std([float(m) for m in newt]))
        
#for k in mydict.keys():
#    print(np.min(mydict[k]))
with open('data/prep.csv', 'a') as myfile:
    for i in mydict.keys():
        myfile.write(i + ',' + str(np.mean(mydict[i])) + ',' + str(np.mean(mydictstd[i])) + '\n') 
    