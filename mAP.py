"""
compute mAP given a result file
"""
import glob, os
import re
import numpy as np
import  matplotlib.pyplot as plt
regex = ''

def get_map(file):
    mAP = 0.0
    i = 0

    with open(file) as f:        
        for line in f :
            result = line.strip().split(',')
            ranking = np.array([int(int(elem) == int(result[0])) for elem in result[1:]])
            inds = np.arange(1, len(ranking)+1)
            recall = np.cumsum(ranking) * ranking
            valid_inds = recall != 0
            inds = inds[valid_inds]
            recall = recall[valid_inds]
            precision = recall / inds
            AP = np.mean(precision)
            mAP += AP
            i +=1
    return mAP/i
            

if __name__ == '__main__':
    folder = os.path.join(os.getcwd(), 'results')
    dmAP = {}
    names = []
    colors = []

    for idx, file in enumerate(sorted(glob.glob(os.path.join(folder, "*.csv")))):
        name_file = file.split('/')[-1]
        names.append(name_file.split("_")[0].capitalize() + " encoder \n" + name_file.split("_")[-1].split(".")[0].capitalize())
        dmAP[idx] = get_map(file)

        # Determinar el color en función del nombre del archivo
        if name_file.startswith("two"):
            colors.append('red')
        elif name_file.startswith("one"):
            colors.append('blue')
        else:
            colors.append('gray')  # Color por defecto para otros archivos

    keys = list(dmAP.keys())
    keys = sorted(keys)
    its = []
    mAPs = []

    for it in keys:
        print('{}: {}'.format(names[it], dmAP[it]), flush=True)
        its.append(it)
        mAPs.append(dmAP[it])

    plt.bar(names, mAPs, color=colors)
    
    # Agregar una barra adicional en el medio con un valor de mAP de 0
    plt.bar(0.5, 0, color='white', width=0.5)
    plt.title('mAP')

    # Agregar el valor de mAP dentro de cada barra
    for i, mAP in enumerate(mAPs):
        plt.text(i, mAP, f'{mAP:.2f}', ha='center', va='bottom')
    
    plt.show()