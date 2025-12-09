import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import random
import math

class PointCoverage:
    def __init__(self, field_size=100, radius=12, max_points=25):
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
    
    # region Начальное решение
    def initialize_points(self, num_points):
        """Инициализация начального расположения точек (центры только внутри поля)"""
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

    # region Метод Монте-Карло
    def calculate_coverage_with_overlap(self):
        """Вычисление покрытой площади методом Монте-Карло (учитывается только поле)"""
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
    
    # region Целевая функция
    def objective_function(self):
        """Целевая функция: только покрытие поля"""
        return self.calculate_coverage_with_overlap()
    
    def simulated_annealing(self, initial_temp=10000, cooling_rate=0.999999, 
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
            
            # Прогресс
            if best_coverage<new_coverage:
                print(f"Итерация {iteration}, "
                      f"Лучшее: {best_coverage*100:.1f}%")
            
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
        
        self.points = best_solution
        final_coverage = best_coverage
        
        print(f"\nОптимизация завершена!")
        print(f"Лучшее покрытие: {final_coverage*100:.2f}%")
        print(f"Количество точек: {len(best_solution)}")
        
        return best_solution, final_coverage
    
    # region Новое решение
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
                # Случайное смещение (центры остаются внутри поля)
                dx = random.uniform(-self.radius, self.radius)
                dy = random.uniform(-self.radius, self.radius)
                
                new_x = max(0, min(self.field_size, x + dx))
                new_y = max(0, min(self.field_size, y + dy))
                
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
                # Центры только внутри поля
                x = random.uniform(0, self.field_size)
                y = random.uniform(0, self.field_size)
                
                if self.is_valid_position(x, y, new_solution):
                    new_solution.append([x, y])
                    added = True
                
                attempts += 1
        
        elif operation == 'remove' and len(new_solution) > 1:
            # Удаление случайной точки
            idx = random.randint(0, len(new_solution) - 1)
            new_solution.pop(idx)
        
        return new_solution