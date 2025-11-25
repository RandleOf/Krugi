import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import math
from PointCoverage_cl import PointCoverage

def animate_optimization(field_size=100, radius=15, max_points=25):
    """Создание анимации процесса оптимизации"""
    
    # Создание объекта покрытия
    coverage = PointCoverage(field_size, radius, max_points)
    
    # Инициализация с небольшим количеством точек
    initial_points = 5
    coverage.initialize_points(initial_points)
    
    # Запуск имитации отжига
    best_solution, best_coverage = coverage.simulated_annealing(
        initial_temp=10000, cooling_rate=0.99, max_iterations=10000
    )
    
    # Настройка анимации
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    def update(frame):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Отображение текущего состояния
        current_solution = coverage.solution_history[frame]
        current_coverage = coverage.coverage_history[frame] * 100
        best_coverage_current = coverage.best_coverage_history[frame] * 100
        
        # График 1: Покрытие поля
        ax1.set_xlim(-radius, field_size + radius)
        ax1.set_ylim(-radius, field_size + radius)
        ax1.set_title(f'Шаг {frame}: {current_coverage:.1f}% покрытия\n(Лучшее: {best_coverage_current:.1f}%)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal') 
        
        # Отображение поля
        field_rect = plt.Rectangle((0, 0), field_size, field_size, 
                                 fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(field_rect)
        
        # Отображение точек и их зон покрытия
        colors = plt.cm.viridis(np.linspace(0, 1, len(current_solution)))
        for idx, (x, y) in enumerate(current_solution):
            circle = Circle((x, y), radius, fill=True, alpha=0.2, color=colors[idx])
            ax1.add_patch(circle)
            ax1.plot(x, y, 'o', markersize=5, color=colors[idx], 
                    markeredgecolor='black', markeredgewidth=1)
            
            # Отображение минимального расстояния (радиус) между точками
            for j, (x2, y2) in enumerate(current_solution):
                if idx < j:  # Чтобы не рисовать линии дважды
                    distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < radius * 2:  # Показываем линии для близких точек
                        ax1.plot([x, x2], [y, y2], 'gray', alpha=0.3, linewidth=1)
        
        # График 2: Эволюция покрытия
        ax2.plot([c*100 for c in coverage.coverage_history[:frame+1]], 'b-', alpha=0.7, label='Текущее')
        ax2.plot([c*100 for c in coverage.best_coverage_history[:frame+1]], 'r-', linewidth=2, label='Лучшее')
        ax2.set_title('Эволюция покрытия поля')
        ax2.set_xlabel('Итерация')
        ax2.set_ylabel('Покрытие (%)')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Эволюция энергии (покрытия)
        ax3.plot(coverage.energy_history[:frame+1], 'g-', linewidth=2)
        ax3.set_title('Эволюция целевой функции')
        ax3.set_xlabel('Итерация')
        ax3.set_ylabel('Покрытие')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # График 4: Количество точек
        point_counts = [len(sol) for sol in coverage.solution_history[:frame+1]]
        ax4.plot(point_counts, 'purple', linewidth=2)
        ax4.set_title('Количество точек')
        ax4.set_xlabel('Итерация')
        ax4.set_ylabel('Точек')
        ax4.set_ylim(0, max_points)
        ax4.grid(True, alpha=0.3)
        
        # Информация
        info_text = f'Точек: {len(current_solution)}\nПокрытие: {current_coverage:.1f}%\nЛучшее: {best_coverage_current:.1f}%'
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Создание анимации
    anim = FuncAnimation(fig, update, frames=len(coverage.solution_history), 
                        interval=200, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    # Визуализация финального результата
    final_coverage = best_coverage * 100
    
    plt.figure(figsize=(15, 5))
    
    # Левый график - финальное расположение
    plt.subplot(1, 3, 1)
    plt.xlim(-radius, field_size + radius)
    plt.ylim(-radius, field_size + radius)
    plt.title(f'ФИНАЛЬНОЕ РЕШЕНИЕ\n{len(best_solution)} точек\n{final_coverage:.1f}% покрытия')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    
    # Поле
    field_rect = plt.Rectangle((0, 0), field_size, field_size, 
                             fill=False, edgecolor='black', linewidth=2)
    plt.gca().add_patch(field_rect)
    
    # Точки и зоны покрытия
    colors = plt.cm.viridis(np.linspace(0, 1, len(best_solution)))
    for idx, (x, y) in enumerate(best_solution):
        circle = Circle((x, y), radius, fill=True, alpha=0.25, color=colors[idx])
        plt.gca().add_patch(circle)
        plt.plot(x, y, 'o', markersize=6, color=colors[idx], 
                markeredgecolor='black', markeredgewidth=1)
        
        # Отображение минимального расстояния между точками
        for j, (x2, y2) in enumerate(best_solution):
            if idx < j:
                distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < radius * 2:  # Показываем линии для близких точек
                    plt.plot([x, x2], [y, y2], 'gray', alpha=0.3, linewidth=1)
    
    # Средний график - история лучшего покрытия
    plt.subplot(1, 3, 2)
    plt.plot([c*100 for c in coverage.best_coverage_history], 'r-', linewidth=2)
    plt.title('История лучшего покрытия')
    plt.xlabel('Итерация')
    plt.ylabel('Покрытие (%)')
    plt.grid(True, alpha=0.3)
    
    # Правый график - метрики эффективности
    plt.subplot(1, 3, 3)
    efficiency = final_coverage / len(best_solution) if len(best_solution) > 0 else 0
    
    # Проверка минимального расстояния
    min_distance = float('inf')
    for i, (x1, y1) in enumerate(best_solution):
        for j, (x2, y2) in enumerate(best_solution):
            if i < j:
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                min_distance = min(min_distance, distance)
    
    # Проверка, что все центры внутри поля
    all_centers_inside = all(0 <= x <= field_size and 0 <= y <= field_size for x, y in best_solution)
    
    metrics = ['Покрытие', 'Точки', 'Эффективность', 'Мин. расстояние']
    values = [final_coverage, len(best_solution), efficiency, min_distance]
    colors = ['green', 'blue', 'orange', 'red']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Финальные метрики')
    plt.ylabel('Значение')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, values):
        if bar.get_height() >= 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    f'{value:.1f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ===")
    print(f"Лучшее покрытие: {final_coverage:.2f}%")
    print(f"Количество точек: {len(best_solution)}")
    print(f"Эффективность: {efficiency:.2f}% покрытия на точку")
    print(f"Минимальное расстояние между центрами: {min_distance:.2f} (радиус = {radius})")
    
    # Проверка соблюдения ограничений
    if min_distance >= radius:
        print("✓ Ограничение минимального расстояния выполнено!")
    else:
        print("✗ Ограничение минимального расстояния не выполнено!")
    
    if all_centers_inside:
        print("✓ Все центры окружностей находятся внутри поля!")
    else:
        print("✗ Некоторые центры окружностей находятся вне поля!")
    
    # Вывод координат точек
    print(f"\nКоординаты точек (центры внутри поля):")
    for i, (x, y) in enumerate(best_solution):
        print(f"Точка {i+1}: ({x:.2f}, {y:.2f})")