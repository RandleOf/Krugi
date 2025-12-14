import pygame
import numpy as np

def create_binary_matrix():
    """
    Функция создает интерактивное окно для рисования бинарной матрицы 100x100.
    Возвращает массив NumPy размером 100x100 со значениями 0 и 1.
    """
    # Инициализация Pygame
    pygame.init()
    
    # Размеры окна
    CELL_SIZE = 8
    GRID_SIZE = 100
    WIDTH = GRID_SIZE * CELL_SIZE + 200  # Дополнительное место для панели
    HEIGHT = GRID_SIZE * CELL_SIZE
    
    # Цвета
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)
    BLUE = (70, 130, 180)
    RED = (220, 60, 60)
    
    # Создание окна
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Создание бинарной матрицы 100x100")
    
    # Создание матрицы 100x100 с нулями (все клетки белые)
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    # Переменные для кисти
    brush_size = 1  # Размер кисти (1x1, 3x3, 5x5)
    drawing = False  # Флаг рисования
    erasing = False  # Флаг стирания
    
    # Основной цикл игры
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        # Проверяем, находится ли курсор над панелью инструментов
        mouse_over_panel = mouse_pos[0] > GRID_SIZE * CELL_SIZE
        
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Нажатие кнопки мыши
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                
                # Проверяем, не нажали ли мы на панель инструментов
                if x > GRID_SIZE * CELL_SIZE:
                    # Определяем, на какую кнопку нажали
                    panel_x = GRID_SIZE * CELL_SIZE
                    if panel_x + 20 <= x <= panel_x + 180:
                        # Кнопка размера кисти 1
                        if 50 <= y <= 90:
                            brush_size = 1
                        # Кнопка размера кисти 3
                        elif 100 <= y <= 140:
                            brush_size = 3
                        # Кнопка размера кисти 5
                        elif 150 <= y <= 190:
                            brush_size = 5
                        # Кнопка сохранения и выхода
                        elif 250 <= y <= 290:
                            running = False
                else:
                    # Нажатие на поле для рисования
                    if event.button == 1:  # Левая кнопка мыши
                        drawing = True
                        erasing = False
                    elif event.button == 3:  # Правая кнопка мыши
                        erasing = True
                        drawing = False
            
            # Отпускание кнопки мыши
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                erasing = False
            
            # Движение мыши с зажатой кнопкой
            elif event.type == pygame.MOUSEMOTION:
                if (drawing or erasing) and not mouse_over_panel:
                    x, y = event.pos
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    
                    # Определяем область для рисования/стирания в зависимости от размера кисти
                    half_size = brush_size // 2
                    
                    for dy in range(-half_size, half_size + 1):
                        for dx in range(-half_size, half_size + 1):
                            nx, ny = grid_x + dx, grid_y + dy
                            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                                if drawing:
                                    matrix[ny][nx] = 1  # Рисуем черным
                                elif erasing:
                                    matrix[ny][nx] = 0  # Стираем (делаем белым)
        
        # Отрисовка
        screen.fill(WHITE)
        
        # Рисуем клетки
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = WHITE if matrix[y][x] == 0 else BLACK
                pygame.draw.rect(screen, color, 
                                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, GRAY, 
                                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        
        # Рисуем панель инструментов
        panel_x = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(screen, (240, 240, 240), (panel_x, 0, 200, HEIGHT))
        pygame.draw.line(screen, GRAY, (panel_x, 0), (panel_x, HEIGHT), 2)
        
        # Заголовок панели
        font = pygame.font.SysFont(None, 30)
        title = font.render("Инструменты", True, BLUE)
        screen.blit(title, (panel_x + 50, 10))
        
        # Кнопки выбора размера кисти
        font = pygame.font.SysFont(None, 24)
        
        # Кисть 1x1
        color = RED if brush_size == 1 else BLUE
        pygame.draw.rect(screen, color, (panel_x + 20, 50, 160, 40), border_radius=5)
        pygame.draw.rect(screen, BLACK, (panel_x + 20, 50, 160, 40), 2, border_radius=5)
        text1 = font.render("Кисть 1x1", True, WHITE)
        screen.blit(text1, (panel_x + 65, 62))
        
        # Кисть 3x3
        color = RED if brush_size == 3 else BLUE
        pygame.draw.rect(screen, color, (panel_x + 20, 100, 160, 40), border_radius=5)
        pygame.draw.rect(screen, BLACK, (panel_x + 20, 100, 160, 40), 2, border_radius=5)
        text2 = font.render("Кисть 3x3", True, WHITE)
        screen.blit(text2, (panel_x + 65, 112))
        
        # Кисть 5x5
        color = RED if brush_size == 5 else BLUE
        pygame.draw.rect(screen, color, (panel_x + 20, 150, 160, 40), border_radius=5)
        pygame.draw.rect(screen, BLACK, (panel_x + 20, 150, 160, 40), 2, border_radius=5)
        text3 = font.render("Кисть 5x5", True, WHITE)
        screen.blit(text3, (panel_x + 65, 162))
        
        # Инструкция
        instr_font = pygame.font.SysFont(None, 20)
        instr1 = instr_font.render("ЛКМ - рисовать черным", True, BLACK)
        instr2 = instr_font.render("ПКМ - стирать (белым)", True, BLACK)
        instr3 = instr_font.render("Зажмите для рисования", True, BLACK)
        
        screen.blit(instr1, (panel_x + 30, 210))
        screen.blit(instr2, (panel_x + 30, 230))
        screen.blit(instr3, (panel_x + 30, 250))
        
        # Кнопка сохранения
        pygame.draw.rect(screen, (60, 180, 110), (panel_x + 20, 250, 160, 40), border_radius=5)
        pygame.draw.rect(screen, BLACK, (panel_x + 20, 250, 160, 40), 2, border_radius=5)
        save_text = font.render("Сохранить", True, WHITE)
        screen.blit(save_text, (panel_x + 65, 262))
        
        # Отображение текущего размера кисти
        brush_text = font.render(f"Текущая кисть: {brush_size}x{brush_size}", True, BLACK)
        screen.blit(brush_text, (panel_x + 30, 310))
        
        # Статистика
        black_cells = np.sum(matrix)
        stats_text1 = font.render(f"Черных: {black_cells}", True, BLACK)
        stats_text2 = font.render(f"Белых: {GRID_SIZE*GRID_SIZE - black_cells}", True, BLACK)
        stats_text3 = font.render(f"Всего: {GRID_SIZE*GRID_SIZE}", True, BLACK)
        
        screen.blit(stats_text1, (panel_x + 30, 350))
        screen.blit(stats_text2, (panel_x + 30, 380))
        screen.blit(stats_text3, (panel_x + 30, 410))
        
        # Предварительный просмотр кисти (если курсор над полем)
        if not mouse_over_panel and mouse_pos[0] < panel_x:
            x, y = mouse_pos
            grid_x = x // CELL_SIZE
            grid_y = y // CELL_SIZE
            
            half_size = brush_size // 2
            
            for dy in range(-half_size, half_size + 1):
                for dx in range(-half_size, half_size + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        # Рисуем полупрозрачную подсветку
                        preview_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        preview_color = (255, 0, 0, 100) if drawing or not (drawing or erasing) else (0, 0, 255, 100)
                        preview_surface.fill(preview_color)
                        screen.blit(preview_surface, (nx * CELL_SIZE, ny * CELL_SIZE))
        
        pygame.display.flip()
    
    pygame.quit()
    
    # Возвращаем созданную матрицу
    return matrix