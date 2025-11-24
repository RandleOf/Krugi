import matplotlib.pyplot as plt
import numpy as np
import random

#проверка на финальную позицию
def finish(posm, final_pos): 
    if posm[0] == final_pos[0] and posm[1] == final_pos[1]:
        return False
    return True 

#создание матрици xmax на ymax членов
def mat(xmax, ymax, c): 
    lok = np.zeros((xmax, ymax))
    for i in range(ymax):
        for j in range(xmax):
            lok[j,i]=c
    return lok

#создание поля координат
def koord(xmax, ymax):
    pos = np.empty((xmax, ymax), dtype=object)
    for i in range(ymax):
        for j in range(xmax):
            pos[j, i]=np.array([j, i])
    return pos

#проверка выхода за край
def kletki(posm, xmax, ymax, steni):
    sosedi = []
    x = posm[0]
    y = posm[1]
    
    # Все 8 возможных направлений (включая диагонали)
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),    # горизонталь/вертикаль
        (1, 1), (1, -1), (-1, 1), (-1, -1)   # диагонали
    ]
    
    for dx, dy in directions:
        new_x = x + dx
        new_y = y + dy
        
        # Проверяем границы
        if 0 <= new_x < xmax and 0 <= new_y < ymax:
            # Проверяем, нет ли стены
            if steni[new_x, new_y] == 0:
                sosedi.append(np.array([new_x, new_y]))
    
    return sosedi

def isparenie(fer, xmax, ymax):
    for i in range(xmax):
        for j in range(ymax):
            fer[i, j] = fer[i, j]*0.7
    return fer

def osnova(fer, posm, final_pos, l, m, steni, xmax, ymax):
    ver0 = []
    ver = []
    sum = 0
    for i in kletki(posm, xmax, ymax, steni):
        x = i[0]
        y = i[1]
        a = fer[x, y]
        dist = abs(i[0] - final_pos[0]) + abs(i[1] - final_pos[1]) + 0.1
        b = 1/dist
        ver0.append((a**l)*(b**m))
    for i in range(len(ver0)):
        sum += ver0[i]
    for i in range(len(ver0)):
        ver.append(ver0[i]/sum)
    return ver

def kletka(fer, posm, final_pos, l, m, steni, xmax, ymax):
    os = osnova(fer, posm, final_pos, l, m, steni, xmax, ymax)
    x = 0                                
    p = random.random()             
    j = 0 
    posnew = kletki(posm, xmax, ymax, steni)
    current = 0
    for i, length in enumerate(os, 1):
        if current <= p < current + length:
            j = i-1
            break
        current += length
    return posnew[j]

def minsind(mass):
    min_val = max(mass)
    ind = 0
    for i in range(len(mass)):
        if min_val > mass[i]:
            min_val = mass[i]
            ind = i
    return ind

start_pos = np.array([0, 50])
final_pos = np.array([50, 0])
xmax = 100
ymax = 100
l = 1
m = 10

vseputi = []
pos = koord(xmax, ymax)
lok = mat(xmax, ymax, 0)
fer = mat(xmax, ymax, 0.2)

vsepyti = []
vsedlini = []
for iteration in range(10):
    dlinam = []
    posm = start_pos
    position = []
    for j in range(50):
        dlina = 0
        posmas = []
        posm = start_pos
        while finish(posm, final_pos):
            posmas.append(posm)
            posm = kletka(fer, posm, final_pos, l, m, lok, xmax, ymax)
            dlina += 1
            print(posm)
        dlinam.append(dlina)
        position.append(posmas)
        print('муравей готов')
    vsedlini.append(min(dlinam))
    vsepyti.append(position[minsind(dlinam)])
    fer = isparenie(fer, xmax, ymax)
    for path in position:
        for koord in path:  # ИСПРАВЛЕНИЕ: исправили обращение к координатам
            x = koord[0]
            y = koord[1]
            fer[x, y] += 1/max(dlinam)
    print('итерация готова')

lightmax = min(vsedlini)
rrr = minsind(vsedlini)

pyt = vsepyti[rrr]

pyt2 = []
for i in range(len(pyt)):
    koordd = pyt[i]
    pyt2.append([int(koordd[0]), int(koordd[1])])

x = []
y = []

for i in range(len(pyt)):
    koordd = pyt2[i]
    x0 = koordd[0]
    y0 = koordd[1]
    x.append(x0)
    y.append(y0)
    
plt.xlim(-1 , 100)
plt.ylim(-1 , 100)
plt.grid()
plt.plot(x, y)
plt.show()