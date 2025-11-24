import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import random
import math

class PointCoverage:
    def __init__(self, field_size=100, radius=12, max_points=100):
        self.field_size = field_size
        self.radius = radius
        self.max_points = max_points
        self.points = []
        
    def is_valid_position(self, x, y, points_list=None):
        """Проверка, что новая точка находится на достаточном расстоянии от всех существующих"""
        if points_list is None:
            points_list = self.points
            
        for px, py in points_list:
            distance = math.sqrt((x - px)**2 + (y - py)**2)
            if distance < self.radius:  # Минимальное расстояние = радиус
                return False
        return True
    
    def initialize_points(self, num_points):
        """Инициализация начального расположения точек с учетом минимального расстояния"""
        self.points = []
        attempts = 0
        max_attempts = 1000
        
        while len(self.points) < num_points and attempts < max_attempts:
            # Центры только внутри поля
            x = random.uniform(0, self.field_size)
            y = random.uniform(0, self.field_size)
            
            if self.is_valid_position(x, y):
                self.points.append([x, y])
            
            attempts += 1
        
        if len(self.points) < num_points:
            print(f"Предупреждение: удалось разместить только {len(self.points)} из {num_points} точек")
    
    def calculate_coverage_with_overlap(self):
        """Вычисление покрытой площади методом Монте-Карло"""
        if not self.points:
            return 0
        
        num_samples = 5000
        covered_samples = 0
        
        for _ in range(num_samples):
            x = random.uniform(0, self.field_size)
            y = random.uniform(0, self.field_size)
            
            for px, py in self.points:
                distance = math.sqrt((x - px)**2 + (y - py)**2)
                if distance <= self.radius:
                    covered_samples += 1
                    break
        
        return covered_samples / num_samples
    
    def objective_function(self):
        """Целевая функция: только покрытие поля"""
        return self.calculate_coverage_with_overlap()
    
    def simulated_annealing(self, initial_temp=10000, cooling_rate=0.98, 
                           min_temp=0.01, max_iterations=10000):
        """Метод имитации отжига для максимизации покрытия с учетом минимального расстояния"""
        current_solution = self.points.copy()
        self.points = current_solution
        current_coverage = self.calculate_coverage_with_overlap()
        current_energy = self.objective_function()
        
        best_solution = current_solution.copy()
        best_coverage = current_coverage
        best_energy = current_energy
        
        temperature = initial_temp
        iteration = 0
        
        # Для анимации
        self.energy_history = [current_energy]
        self.coverage_history = [current_coverage]
        self.solution_history = [current_solution.copy()]
        self.best_coverage_history = [current_coverage]
        
        print("Запуск оптимизации покрытия поля...")
        print(f"Начальное покрытие: {current_coverage*100:.2f}%")
        
        while temperature > min_temp and iteration < max_iterations:
            # Создание нового решения
            new_solution = self.generate_neighbor(current_solution)
            self.points = new_solution
            new_coverage = self.calculate_coverage_with_overlap()
            new_energy = self.objective_function()
            
            # Принятие решения по методу отжига
            energy_difference = new_energy - current_energy
            
            if energy_difference > 0 or random.random() < math.exp(energy_difference / temperature):
                current_solution = new_solution
                current_energy = new_energy
                current_coverage = new_coverage
            
            # Сохранение лучшего решения
            if new_coverage > best_coverage:
                best_solution = new_solution.copy()
                best_coverage = new_coverage
                best_energy = new_energy
            
            # Сохранение для анимации
            self.energy_history.append(current_energy)
            self.coverage_history.append(current_coverage)
            self.solution_history.append(current_solution.copy())
            self.best_coverage_history.append(best_coverage)
            
            # Охлаждение
            temperature *= cooling_rate
            iteration += 1
            
            # Прогресс
            if iteration % 100 == 0:
                print(f"Итерация {iteration}, Покрытие: {current_coverage*100:.1f}%, "
                      f"Лучшее: {best_coverage*100:.1f}%")
        
        self.points = best_solution
        final_coverage = best_coverage
        
        print(f"\nОптимизация завершена!")
        print(f"Лучшее покрытие: {final_coverage*100:.2f}%")
        print(f"Количество точек: {len(best_solution)}")
        
        return best_solution, final_coverage
    
    def generate_neighbor(self, solution):
        """Генерация соседнего решения для максимизации покрытия с учетом минимального расстояния"""
        new_solution = [point.copy() for point in solution]
        
        # Вероятности операций
        if len(new_solution) <= 2:
            operation_weights = [0.3, 0.6, 0.1]  # move, add, remove
        elif len(new_solution) >= self.max_points:
            operation_weights = [0.7, 0.0, 0.3]  # move, add, remove
        else:
            operation_weights = [0.5, 0.3, 0.2]  # move, add, remove
        
        operation = random.choices(['move', 'add', 'remove'], weights=operation_weights)[0]
        
        if operation == 'move' and new_solution:
            # Перемещение случайной точки с проверкой расстояния
            idx = random.randint(0, len(new_solution) - 1)
            x, y = new_solution[idx]
            
            # Временно удаляем текущую точку для проверки расстояния
            temp_point = new_solution.pop(idx)
            
            attempts = 0
            max_attempts = 50
            moved = False
            
            while not moved and attempts < max_attempts:
                # Случайное смещение
                dx = random.uniform(-self.radius, self.radius)
                dy = random.uniform(-self.radius, self.radius)
                
                new_x = max(self.radius, min(self.field_size - self.radius, x + dx))
                new_y = max(self.radius, min(self.field_size - self.radius, y + dy))
                
                if self.is_valid_position(new_x, new_y, new_solution):
                    new_solution.insert(idx, [new_x, new_y])
                    moved = True
                
                attempts += 1
            
            # Если не удалось переместить, возвращаем исходную точку
            if not moved:
                new_solution.insert(idx, temp_point)
        
        elif operation == 'add' and len(new_solution) < self.max_points:
            # Добавление новой точки в случайную позицию с проверкой расстояния
            attempts = 0
            max_attempts = 100
            added = False
            
            while not added and attempts < max_attempts:
                x = random.uniform(self.radius, self.field_size - self.radius)
                y = random.uniform(self.radius, self.field_size - self.radius)
                
                if self.is_valid_position(x, y, new_solution):
                    new_solution.append([x, y])
                    added = True
                
                attempts += 1
        
        elif operation == 'remove' and len(new_solution) > 1:
            # Удаление случайной точки
            idx = random.randint(0, len(new_solution) - 1)
            new_solution.pop(idx)
        
        return new_solution

def animate_optimization(field_size=100, radius=15, max_points=25):
    """Создание анимации процесса оптимизации"""
    
    # Создание объекта покрытия
    coverage = PointCoverage(field_size, radius, max_points)
    
    # Инициализация с небольшим количеством точек
    initial_points = 5
    coverage.initialize_points(initial_points)
    
    # Запуск имитации отжига
    best_solution, best_coverage = coverage.simulated_annealing(
        initial_temp=10000, cooling_rate=0.98, max_iterations=10000
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
        ax1.set_xlim(0, field_size)
        ax1.set_ylim(0, field_size)
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
    plt.xlim(0, field_size)
    plt.ylim(0, field_size)
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
    
    # Проверка соблюдения ограничения
    if min_distance >= radius:
        print("✓ Ограничение минимального расстояния выполнено!")
    else:
        print("✗ Ограничение минимального расстояния не выполнено!")
    
    # Вывод координат точек
    print(f"\nКоординаты точек (центры на поле):")
    for i, (x, y) in enumerate(best_solution):
        print(f"Точка {i+1}: ({x:.2f}, {y:.2f})")

# Запуск оптимизации с анимацией
if __name__ == "__main__":
    animate_optimization(field_size=100, radius=15, max_points=25)