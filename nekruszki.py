import matplotlib.pyplot as plt
import numpy as np
import random

#region помощь

#создание матрици xmax на ymax членов
def matr(xmax, ymax, c): 
    lok = np.zeros((xmax, ymax))
    for i in range(ymax):
        for j in range(xmax):
            lok[j,i]=c
    return lok


#проверка возможен ли ход
def sosedi(posm, xmax, ymax, steni):
    kletki = []
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
                kletki.append(np.array([new_x, new_y]))
    
    return kletki

#def ygolki(pole, xmax, ymax):
#    ygolkii = []
#    for x in range(xmax):
#        for y in range(ymax):
#            posm = [x, y]
#            sos = sosedi(posm, xmax, ymax, pole)
#            koll = 0
#            for i in range(len(sos)):
#                rar = sos[i]
#                fx = rar[0]
#                fy = rar[1]
#                if pole[fx, fy] == 1:
#                    koll += 1
#            if koll == 1 or koll == 5:
#                ygolkii.append([x, y])
#    return ygolkii

def ygolki(pole, xmax, ymax):
    ygolkii = []
    
    for x in range(xmax):
        for y in range(ymax):
            # Проверяем только клетки со значением 0
            if pole[x][y] == 0:
                # Проверяем, находится ли клетка в углу поля
                is_corner = (x == 0 or x == xmax-1) and (y == 0 or y == ymax-1)
                
                if is_corner:
                    ygolkii.append([x, y])
    
    return ygolkii

def kvadrat(koord1, koord2, lok):
    # ИСПРАВЛЕНИЕ: правильно вычисляем размеры квадрата
    x_start, y_start = koord1
    x_end, y_end = koord2
    
    # Убедимся, что координаты в пределах матрицы
    x_start = max(0, min(x_start, lok.shape[0]-1))
    x_end = max(0, min(x_end, lok.shape[0]-1))
    y_start = max(0, min(y_start, lok.shape[1]-1))
    y_end = max(0, min(y_end, lok.shape[1]-1))
    
    # Заполняем квадрат единицами
    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            lok[i, j] = 1
    return lok


#region основные def

#def massivi(start_pos, pole, xmax, ymax):
#
#    p = [xmax-start_pos[0], start_pos[0], ymax-start_pos[1], start_pos[1]]
#    iteration = max(p)
#    masssivi = []
#    for i in range(iteration):
#        massssivi = []
#        for j in [start_pos[0]-i,start_pos[0]+i]:
#            for l in range(start_pos[1]-i,start_pos[1]+i):
#                if pole[j,l] != 0:
#                    massssivi.append(np.array([j, l]))
#        for j in [start_pos[1]-i,start_pos[1]+i]:
#            for l in range(start_pos[0]-i,start_pos[0]+i):
#                if pole[l,j] != 0:
#                    massssivi.append(np.array([l, j])) 
#        masssivi.append(massssivi)       



#region начальные условия

start_pos = np.array([0,50])
final_pos = np.array([50,0])
xmax = 100
ymax = 100
pole = matr(xmax, ymax, 0)

#region основа
kvadrat([0,0], [30,30], pole)

print(ygolki(pole, xmax, ymax))


    