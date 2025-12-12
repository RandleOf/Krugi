import matplotlib.pyplot as plt
import numpy as np
import random

#region -помощь

def finish(posm, final_pos):
    return not (posm[0] == final_pos[0] and posm[1] == final_pos[1])

#создание матрици xmax на ymax членов
def matr(xmax, ymax, c): 
    return np.full((xmax, ymax), c)

#проверка возможен ли ход
def sosedi(posm, xmax, ymax, steni):
    kletki = []
    x = posm[0]
    y = posm[1]
    
    directions = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1)
    ]
    
    for dx, dy in directions:
        new_x = x + dx
        new_y = y + dy
        
        if 0 <= new_x < xmax and 0 <= new_y < ymax:
            if steni[new_x, new_y] == 0:
                kletki.append(np.array([new_x, new_y]))
    
    return kletki

def sosedi_plus(posm, xmax, ymax):
    kletki = []
    x = posm[0]
    y = posm[1]
    
    directions = [
        (1, 0), (0, -1), 
        (-1, 0), (0, 1) 
    ]
    
    for dx, dy in directions:
        new_x = x + dx
        new_y = y + dy
        
        if 0 <= new_x < xmax and 0 <= new_y < ymax:
            kletki.append(np.array([new_x, new_y]))
    
    return kletki

def sososedi(posm, xmax, ymax, steni):
    kletki = []
    x = posm[0]
    y = posm[1]
    
    directions = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1)
    ]
    
    for dx, dy in directions:
        new_x = x + dx
        new_y = y + dy
        
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
                    xx = int(i[0])
                    yy = int(i[1])
                    if pole[xx, yy] == 1:
                        koll+=1
                if koll == 1:
                    ygolki.append([x, y])
    return ygolki

def kvadrat(koord1, koord2, lok):
    x_start, y_start = koord1
    x_end, y_end = koord2
    
    x_start = max(0, min(x_start, lok.shape[0]-1))
    x_end = max(0, min(x_end, lok.shape[0]-1))
    y_start = max(0, min(y_start, lok.shape[1]-1))
    y_end = max(0, min(y_end, lok.shape[1]-1))
    
    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            lok[i, j] = 1
    return lok

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def on_segment(p, q, r):
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
        min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False

def segments_intersect(p1, p2, p3, p4):
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    
    if o4 == 0 and on_segment(p3, p2, p4):
        return True
    
    return False

def mini_otrezki(pole, xmax, ymax):
    otreski = []
    for x in range(xmax):
        for y in range(ymax):
            if pole[x, y] == 1:
                otreski.append([[x, y], [x, y]])
                for sosed in sosedi_plus([x, y], xmax, ymax):
                    xs = int(sosed[0])
                    ys = int(sosed[1])
                    if pole[xs, ys]==1:
                        otreski.append([[x,y], [xs, ys]])
    return otreski

def osnova(fer, final_pos, l, m, mass):
    ver0 = []
    total_sum = 0
    
    for i in mass:
        x = int(i[0])
        y = int(i[1])
        # Используем минимальное значение феромона, если клетка недоступна
        a = max(fer[x, y], 0.001)
        # Эвристика: обратное расстояние до цели
        dist = np.sqrt((x - final_pos[0])**2 + (y - final_pos[1])**2)
        b = 1 / max(dist, 0.001)
        ver0.append((a**l) * (b**m))
    
    total_sum = sum(ver0)
    if total_sum == 0:
        # Возвращаем равномерное распределение
        return [1/len(ver0)] * len(ver0)
    
    # Нормализуем вероятности
    return [v/total_sum for v in ver0]

def kletka(fer, final_pos, l, m, mass):
    if not mass:
        return None
    
    os = osnova(fer, final_pos, l, m, mass)
    
    # Выбираем следующую точку по вероятностям
    r = random.random()
    cumulative = 0
    for i, prob in enumerate(os):
        cumulative += prob
        if r <= cumulative:
            return mass[i]
    
    # Если что-то пошло не так, возвращаем случайную точку
    return random.choice(mass)

def ind(graf, posm):
    for i in range(len(graf)):
        mass = graf[i][0]
        if mass[0] == posm[0] and mass[1] == posm[1]:
            return i
    return -1

#region -основные def

def massiv(start_pos, final_pos, pole, xmax, ymax):
    ugli = ugolki(pole, xmax, ymax)
    otr = mini_otrezki(pole, xmax, ymax)
    sf=[]
    
    # Добавляем старт и финиш в список углов
    ugli.append([start_pos[0], start_pos[1]])
    ugli.append([final_pos[0], final_pos[1]])
    
    # Убираем дубликаты
    ugli = [list(x) for x in set(tuple(x) for x in ugli)]
    
    for ugolos in ugli:
        minimassiv = []
        for ugol in ugli:            
            if ugolos[0] != ugol[0] or ugolos[1] != ugol[1]:
                pravda = True
                for ot in otr:
                    p1ot = ot[0]
                    p2ot = ot[1]
                    if segments_intersect(p1ot, p2ot, ugol, ugolos):                        
                        pravda = False
                        break
                if pravda:
                    minimassiv.append(ugol)
        sf.append([ugolos, minimassiv])
    
    return sf

def maxsind(mass):
    if not mass:
        return 0
    max_val = mass[0]
    max_ind = 0
    for i in range(len(mass)):
        if mass[i] < max_val:  # Ищем минимальное значение
            max_val = mass[i]
            max_ind = i
    return max_ind

def isparenie(fer, xmax, ymax):
    return fer * 0.7

# Функция для визуализации только лучшего результата
def visualize_best_result(pole, start_pos, final_pos, best_path, best_length, best_iteration):
    """Визуализация только лучшего результата"""
    plt.figure(figsize=(15, 5))
    
    # 1. Визуализация лабиринта и лучшего пути
    plt.subplot(1, 3, 1)
    # Отображаем препятствия
    plt.imshow(pole.T, cmap='binary', origin='lower', interpolation='nearest', alpha=0.7)
    
    if best_path and len(best_path) > 1:
        # Преобразуем путь в numpy массив для удобства
        path_array = np.array(best_path)
        # Отображаем путь
        plt.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label=f'Путь ({len(best_path)} точек)')
        # Отмечаем точки пути
        plt.scatter(path_array[:, 0], path_array[:, 1], s=10, c='blue', alpha=0.5)
    
    # Отмечаем старт и финиш
    plt.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Старт', marker='s')
    plt.plot(final_pos[0], final_pos[1], 'ro', markersize=12, label='Финиш', marker='*')
    
    plt.title(f'Лучший путь (Итерация: {best_iteration+1}, Длина: {best_length:.2f})')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right')
    plt.xlim(-1, xmax)
    plt.ylim(-1, ymax)
    
    # 2. Визуализация пути с прогрессией
    plt.subplot(1, 3, 2)
    plt.imshow(pole.T, cmap='binary', origin='lower', interpolation='nearest', alpha=0.7)
    
    if best_path and len(best_path) > 1:
        # Используем цветовую карту для отображения прогресса
        cmap = plt.cm.viridis
        colors = [cmap(i / (len(best_path)-1)) for i in range(len(best_path)-1)]
        
        for i in range(len(best_path)-1):
            plt.plot([best_path[i][0], best_path[i+1][0]], 
                    [best_path[i][1], best_path[i+1][1]], 
                    color=colors[i], linewidth=2, alpha=0.8)
    
    plt.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Старт', marker='s')
    plt.plot(final_pos[0], final_pos[1], 'ro', markersize=12, label='Финиш', marker='*')
    
    plt.title('Путь с цветовой прогрессией')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(-1, xmax)
    plt.ylim(-1, ymax)
    
    # 3. График расстояний между точками
    plt.subplot(1, 3, 3)
    if best_path and len(best_path) > 1:
        distances = []
        for i in range(1, len(best_path)):
            dist = np.sqrt((best_path[i][0] - best_path[i-1][0])**2 + 
                          (best_path[i][1] - best_path[i-1][1])**2)
            distances.append(dist)
        
        x_points = range(1, len(best_path))
        plt.bar(x_points, distances, alpha=0.6, color='skyblue', label='Расстояние между точками')
        plt.plot(x_points, distances, 'r-', linewidth=2, marker='o', markersize=4, label='Тренд')
        
        plt.xlabel('Номер шага')
        plt.ylabel('Расстояние')
        plt.title(f'Распределение расстояний\nВсего шагов: {len(best_path)-1}')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Выводим информацию о пути
    print("\n" + "="*60)
    print("ИНФОРМАЦИЯ О ЛУЧШЕМ ПУТИ")
    print("="*60)
    if best_path and len(best_path) > 0:
        print(f"Итерация: {best_iteration + 1}")
        print(f"Длина пути: {best_length:.2f}")
        print(f"Количество точек в пути: {len(best_path)}")
        print(f"Старт: {best_path[0]}")
        print(f"Финиш: {best_path[-1]}")
        
        if len(best_path) > 1:
            # Вычисляем статистику
            total_dist = sum(np.sqrt((best_path[i][0] - best_path[i-1][0])**2 + 
                                    (best_path[i][1] - best_path[i-1][1])**2) 
                           for i in range(1, len(best_path)))
            print(f"Суммарное расстояние (проверка): {total_dist:.2f}")
            
            # Прямолинейное расстояние
            straight_dist = np.sqrt((best_path[-1][0] - best_path[0][0])**2 + 
                                   (best_path[-1][1] - best_path[0][1])**2)
            efficiency = (straight_dist / best_length * 100) if best_length > 0 else 0
            print(f"Прямолинейное расстояние: {straight_dist:.2f}")
            print(f"Эффективность: {efficiency:.1f}%")

#region -начальные условия
start_pos = np.array([50, 0])
final_pos = np.array([60, 75])
xmax = 100
ymax = 100
pole = matr(xmax, ymax, 0)
kolit = 100  # количество итераций
kolmy = 150  # количество муравьев на итерацию

# Создаем препятствие (квадрат)
kvadrat([90, 10], [95, 95], pole)
kvadrat([5, 5], [95, 15], pole)
kvadrat([5, 60], [40, 61], pole)
kvadrat([5, 30], [10, 95], pole)
kvadrat([5, 90], [95, 95], pole)
kvadrat([65, 35], [70, 80], pole)
kvadrat([39, 35], [40, 60], pole)
kvadrat([39, 35], [70, 36], pole)
kvadrat([20, 79], [70, 80], pole)

# Инициализируем феромоны
fer = matr(xmax, ymax, 0.1)  # начальный уровень феромонов
l = 1  # вес феромонов
m = 2  # вес эвристики

# Устанавливаем высокий уровень феромонов в стартовой точке
fer[start_pos[0], start_pos[1]] = 1.0

#region -основа
print("Начинаем работу муравьиного алгоритма...")
print(f"Старт: {start_pos}, Финиш: {final_pos}")
print(f"Размер поля: {xmax}x{ymax}")
print(f"Количество итераций: {kolit}, Муравьев на итерацию: {kolmy}")

# Строим граф возможных перемещений
print("\nСтроим граф возможных перемещений...")
graf = massiv(start_pos, final_pos, pole, xmax, ymax)
print(f"Построено {len(graf)} узлов в графе")

vsepyti = []  # лучшие пути на каждой итерации
vsedlini = []  # длины лучших путей

# Основной цикл алгоритма
for iteration in range(kolit):
    print(f"\n--- Итерация {iteration+1}/{kolit} ---")
    
    dlinam = []  # длины путей муравьев
    position = []  # пути муравьев
    
    # Запускаем всех муравьев
    for j in range(kolmy):
        dlina = 0
        posmas = []
        posm = start_pos.copy()  # начинаем со старта
        
        # Муравей идет к цели
        steps = 0
        max_steps = 500  # максимальное количество шагов
        
        while finish(posm, final_pos) and steps < max_steps:
            posmas.append(posm.copy())
            index = ind(graf, posm)
            
            if index == -1:
                print(f"Муравей {j+1}: позиция {posm} не найдена в графе")
                break
            
            graf0 = graf[index]
            next_point = kletka(fer, final_pos, l, m, graf0[1])
            
            if next_point is None:
                print(f"Муравей {j+1}: не удалось выбрать следующую точку")
                break
            
            # Вычисляем расстояние
            dist = np.sqrt((next_point[0] - posm[0])**2 + (next_point[1] - posm[1])**2)
            dlina += dist
            
            # Переходим к следующей точке
            posm = np.array(next_point)
            steps += 1
        
        # Если достигли цели, добавляем финальную точку
        if not finish(posm, final_pos):
            posmas.append(posm.copy())
        
        # Сохраняем путь и длину
        dlinam.append(dlina)
        position.append(posmas)
        
        if len(posmas) > 0 and not finish(posmas[-1], final_pos):
            print(f"Муравей {j+1}: достиг цели за {len(posmas)} шагов, длина: {dlina:.2f}")
        else:
            print(f"Муравей {j+1}: не достиг цели, шагов: {len(posmas)}, длина: {dlina:.2f}")
    
    # Сохраняем лучший путь на этой итерации
    if dlinam:
        # Ищем минимальную длину (лучший путь)
        min_dlina = min(dlinam)
        min_index = np.argmin(dlinam)
        
        vsedlini.append(min_dlina)
        vsepyti.append(position[min_index])
        
        print(f"Лучший путь на итерации: длина = {min_dlina:.2f}, шагов = {len(position[min_index])}")
    else:
        vsedlini.append(0)
        vsepyti.append([])
        print("На этой итерации пути не найдены")
    
    # Испаряем феромоны
    fer = isparenie(fer, xmax, ymax)
    
    # Обновляем феромоны на путях муравьев
    for path_idx, path in enumerate(position):
        if path and dlinam[path_idx] > 0:
            # Количество феромонов обратно пропорционально длине пути
            pheromone_amount = 10.0 / dlinam[path_idx]
            
            for koord in path:
                x = int(koord[0])
                y = int(koord[1])
                if 0 <= x < xmax and 0 <= y < ymax:
                    fer[x, y] += pheromone_amount

print("\n" + "="*60)
print("АЛГОРИТМ ЗАВЕРШЕН")
print("="*60)

# Находим и отображаем лучший результат
if vsedlini:
    # Ищем ненулевые длины
    nonzero_indices = [i for i, dlina in enumerate(vsedlini) if dlina > 0]
    
    if nonzero_indices:
        # Находим минимальную длину среди ненулевых
        valid_lengths = [vsedlini[i] for i in nonzero_indices]
        best_valid_idx = np.argmin(valid_lengths)
        best_iteration = nonzero_indices[best_valid_idx]
        best_length = valid_lengths[best_valid_idx]
        best_path = vsepyti[best_iteration]
        
        print(f"\nЛУЧШИЙ РЕЗУЛЬТАТ:")
        print(f"Итерация: {best_iteration + 1}")
        print(f"Длина пути: {best_length:.2f}")
        
        if best_path and len(best_path) > 0:
            print(f"Точек в пути: {len(best_path)}")
            print(f"Первый шаг: {best_path[0]}")
            print(f"Последний шаг: {best_path[-1]}")
            
            # Визуализируем лучший результат
            visualize_best_result(pole, start_pos, final_pos, best_path, best_length, best_iteration)
        else:
            print("Путь пустой, визуализация невозможна")
    else:
        print("Все пути имеют нулевую длину")
        
        # Покажем хотя бы лабиринт
        plt.figure(figsize=(8, 8))
        plt.imshow(pole.T, cmap='binary', origin='lower', interpolation='nearest', alpha=0.7)
        plt.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Старт', marker='s')
        plt.plot(final_pos[0], final_pos[1], 'ro', markersize=12, label='Финиш', marker='*')
        plt.title('Лабиринт (пути не найдены)')
        plt.xlabel('X координата')
        plt.ylabel('Y координата')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.xlim(-1, xmax)
        plt.ylim(-1, ymax)
        plt.tight_layout()
        plt.show()
else:
    print("Нет данных о путях")

print("\nРабота программы завершена!")