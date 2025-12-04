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
        (1, 0), (1, -1), (0, -1), (-1, -1),    # горизонталь/вертикаль
        (-1, 0), (-1, 1), (0, 1), (1, 1)   # диагонали
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

def sososedi(posm, xmax, ymax, steni):
    kletki = []
    x = posm[0]
    y = posm[1]
    
    # Все 8 возможных направлений (включая диагонали)
    directions = [
        (1, 0), (1, -1), (0, -1), (-1, -1),    # горизонталь/вертикаль
        (-1, 0), (-1, 1), (0, 1), (1, 1)   # диагонали
    ]
    
    for dx, dy in directions:
        new_x = x + dx
        new_y = y + dy
        
        # Проверяем границы
        if 0 <= new_x < xmax and 0 <= new_y < ymax:
            kletki.append(np.array([new_x, new_y]))
    
    return kletki

def ugolki(pole, xmax, ymax):
    ygolki = []
    for x in range(xmax):
        for y in range(ymax):
            posm = [x, y]
            soss = sososedi(posm, xmax, ymax, pole)
            koll = 0
            if pole[x, y]==0:
                for i in soss:
                    xx = i[0]
                    yy = i[1]

                    if int(pole[xx, yy]) == 1:
                        koll+=1
                if koll == 1:
                    ygolki.append([x, y])



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

def orientation(p, q, r):
    """Возвращает ориентацию тройки точек (p, q, r)
    Возвращаемые значения:
    0 - точки коллинеарны
    1 - по часовой стрелке
    2 - против часовой стрелки
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if val == 0:
        return 0  # коллинеарны
    elif val > 0:
        return 1  # по часовой стрелке
    else:
        return 2  # против часовой стрелки

def on_segment(p, q, r):
    """Проверяет, лежит ли точка q на отрезке pr"""
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
        min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False

def segments_intersect(p1, p2, p3, p4):
    """Проверяет, пересекаются ли отрезки (p1, p2) и (p3, p4)
    
    Аргументы:
    p1, p2 - координаты концов первого отрезка (x, y)
    p3, p4 - координаты концов второго отрезка (x, y)
    
    Возвращает:
    True - если отрезки пересекаются
    False - если отрезки не пересекаются
    """
    # Находим ориентации для всех возможных троек
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    # Общий случай (отрезки не коллинеарны)
    if o1 != o2 and o3 != o4:
        return True
    
    # Специальные случаи (коллинеарность)
    # p1, p2 и p3 коллинеарны и p3 лежит на отрезке p1p2
    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    
    # p1, p2 и p4 коллинеарны и p4 лежит на отрезке p1p2
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    
    # p3, p4 и p1 коллинеарны и p1 лежит на отрезке p3p4
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    
    # p3, p4 и p2 коллинеарны и p2 лежит на отрезке p3p4
    if o4 == 0 and on_segment(p3, p2, p4):
        return True
    
    return False

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
kvadrat([0,0], [1,1], pole)




    