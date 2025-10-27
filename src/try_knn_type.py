#!/usr/bin/env python3
"""
generate_features_dataset.py

Генерация синтетических изображений ячеек таблицы и извлечение признаков
для классификации типов содержимого: пустые, слова, числа, один символ.

Признаки:
1. dark_frac — доля чёрных пикселей (плотность содержимого)
2. vert_strokes_count — вертикальные штрихи (цифры имеют больше)
3. horiz_strokes_count — горизонтальные штрихи (таблицы/линии)
4. aspect_ratio — ширина/высота (геометрия содержимого)
5. compactness — заполненность bounding box
6. num_contours — количество контуров (≈ число символов)
7. cc_count — количество связных компонент
8. col_peaks — пики вертикальной проекции (≈ число символов)
9. row_peaks — пики горизонтальной проекции

Requirements:
    pip install numpy opencv-python Pillow pandas matplotlib scipy
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    max_shift = 3
    blur_chance= 0.5
    """
    Simulate low-res, slightly rough/pixelated distortion (like old print or pixel art).

    Args:
        np_img: Input grayscale image (H×W or H×W×1).
        min_scale, max_scale: Downsampling scale range.
        max_shift: Random subpixel translation to make lines less straight.
        morph_kernel_range: Range of kernel sizes for morphological distortion.
        blur_chance: Probability to apply final light blur for realism.

    Returns:
        np.ndarray: Distorted grayscale image.
    """
    h, w = np_img.shape[:2]

    # Step 1: Randomly scale down & up (pixelation)
    scale = random.uniform(0.3, 0.4)
    small = cv2.resize(np_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Step 2: Apply small random affine transform (to bend or jitter lines)
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    up = cv2.warpAffine(up, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    
    up = up.astype(np.float32)
    factor = random.uniform(0.3, 0.4)  # how much to lift dark areas
    mask = up < 100  # only pixels darker than 100 get lifted
    up[mask] = up[mask] + (100 - up[mask]) * factor
    up = np.clip(up, 0, 255).astype(np.uint8)

    # Step 4: Optional soft blur to smooth sharp edges
    if random.random() < blur_chance:
        up = cv2.GaussianBlur(up, (3, 3), sigmaX=random.uniform(1.5, 1.5))

    return up




class CellImageGenerator:
    """Генератор синтетических изображений ячеек таблицы."""
    
    def __init__(self, img_size: Tuple[int, int] = (160,40)):
        self.img_size = img_size  # (width, height)
        self.width, self.height = img_size
        
        # Попытка загрузить русский шрифт
        try:
            self.font_large = ImageFont.truetype("/System/Library/Fonts/Times New Roman.ttf", 13)
            self.font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 13)
        except:
            print("Warning: Could not load DejaVu font, using default")
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
        
        # Русские слова для генерации
        self.russian_words = [
            'Москва', 'Санкт-Петербург', 'Россия', 'январь', 'февраль', 'март',
            'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь',
            'ноябрь', 'декабрь', 'компания', 'продукт', 'услуга', 'клиент',
            'договор', 'итого', 'сумма', 'количество', 'наименование', 'описание',
            'организация', 'предприятие', 'документ', 'отчет', 'баланс', 'прибыль',
            'расход', 'доход', 'задолженность', 'платеж', 'счет', 'накладная',
            'товар', 'цена', 'скидка', 'налог', 'процент', 'ставка', 'курс',
            'акция', 'облигация', 'инвестиция', 'капитал', 'актив', 'пассив'
        ]
        
        self.cyrillic_chars = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    
    def generate_number(self) -> np.ndarray:
        """Генерация изображения с числом."""
        img = Image.new('L', (self.width, self.height), color=255)
        draw = ImageDraw.Draw(img)
        
        # Различные форматы чисел
        patterns = [
            lambda: f"{random.randint(0, 999999):,}".replace(',', ' '),  # 123 456
            lambda: f"{random.uniform(0, 9999):.2f}",  # 123.45
            lambda: f"{random.randint(0, 99999)}",  # 12345
            lambda: f"${random.randint(0, 9999)}",  # $1234
            lambda: f"{random.randint(0, 999)} ₽",  # 123 ₽
            lambda: f"({random.randint(0, 999)})",  # (123)
            lambda: f"-{random.randint(0, 999)}",  # -123
            lambda: f"{random.randint(1, 31)}.{random.randint(1, 12)}.{random.randint(2020, 2025)}",  # дата
            lambda: f"{random.randint(0, 99)}%",  # 50%
            lambda: f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",  # время
        ]
        
        text = random.choice(patterns)()
        font = random.choice([self.font_large, self.font_medium])
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2
        
        draw.text((x, y), text, fill=0, font=font)
        
        np_img=np.array(img).astype(np.float32) / 255.0
        
        np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)

        return np_img
    
    def generate_word(self) -> np.ndarray:
        """Генерация изображения с русским словом или фразой."""
        img = Image.new('L', (self.width, self.height), color=255)
        draw = ImageDraw.Draw(img)
        
        choice = random.choice(['single_word', 'two_words', 'random_text'])
        
        if choice == 'single_word':
            text = random.choice(self.russian_words)
            font = random.choice([self.font_large, self.font_medium])
        elif choice == 'two_words':
            text = f"{random.choice(self.russian_words)} {random.choice(self.russian_words)}"
            font = self.font_small
        else:
            # Случайная последовательность кириллических символов
            length = random.randint(5, 15)
            text = ''.join(random.choice(self.cyrillic_chars) for _ in range(length))
            font = self.font_medium
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = max(5, (self.width - text_width) // 2)
        y = (self.height - text_height) // 2
        draw.text((x, y), text, fill=0, font=font)
        
        np_img=np.array(img).astype(np.float32) / 255.0
        
        np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)

        return np_img
    
    def generate_other(self) -> np.ndarray:
        """Генерация 'других' изображений: пустые, символы, линии."""
        img = Image.new('L', (self.width, self.height), color=255)
        draw = ImageDraw.Draw(img)
        
        choice = random.choice(['empty', 'single_char', 'symbol', 'lines', 'mixed'])
        
        if choice == 'empty':
            # Пустое изображение (иногда с шумом)
            if random.random() < 0.2:
                arr = np.array(img).astype(np.float32) / 255.0
                noise = np.random.normal(0, 0.02, arr.shape)
                arr = np.clip(arr + noise, 0, 1)
                return arr
            return np.array(img).astype(np.float32) / 255.0
        
        elif choice == 'single_char':
            # Один символ (буква или цифра)
            chars = self.cyrillic_chars + '0123456789'
            text = random.choice(chars)
            font = self.font_large
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2
            draw.text((x, y), text, fill=0, font=font)
        
        elif choice == 'symbol':
            # Специальные символы
            symbols = ['-', '—', '•', '*', '/', '\\', '|', '...', '№', '§', '±', '×', '÷']
            text = random.choice(symbols)
            font = random.choice([self.font_large, self.font_medium])
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2
            draw.text((x, y), text, fill=0, font=font)
        
        elif choice == 'lines':
            # Горизонтальные/вертикальные линии
            num_lines = random.randint(1, 3)
            for _ in range(num_lines):
                if random.random() < 0.5:
                    y = random.randint(5, self.height - 5)
                    draw.line([(0, y), (self.width, y)], fill=0, width=random.choice([1, 2]))
                else:
                    x = random.randint(5, self.width - 5)
                    draw.line([(x, 0), (x, self.height)], fill=0, width=random.choice([1, 2]))
        
        elif choice == 'mixed':
            # Смесь символов и цифр
            chars = random.choice([
                f"{random.choice(self.cyrillic_chars)}{random.randint(0, 9)}",
                f"{random.randint(0, 9)}{random.choice(self.cyrillic_chars)}",
                f"№{random.randint(1, 99)}"
            ])
            font = self.font_medium
            bbox = draw.textbbox((0, 0), chars, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2
            draw.text((x, y), chars, fill=0, font=font)
        
        np_img=np.array(img).astype(np.float32) / 255.0
        
        np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)

        return np_img

def calculate_column_intensity(img: np.ndarray, number=1, save_plot=False) -> np.ndarray:
    """
    Calculate average intensity for each column of pixels.
    For text detection, we want high values where there's text content.

    Args:
        img: Grayscale image

    Returns:
        Array of average intensities for each column (normalized 0-1)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to float and normalize to 0-1 range
    img_float = img.astype(np.float32) / 255.0
    
    # convert to 3 np arrays one for top 30% fo rows and one for bottom 30% of rows and one for middle 40% of rows
    h, w = img_float.shape
    top_30 = img_float[:int(h*0.3), :]
    bottom_30=img_float[-int(h*0.3):, :]
    middle_40 = img_float[int(h*0.3):int(h*0.7), :]

    # bottom_30=find_top_character_profile(img_float)
    
    
    max_depth = int(h* 0.3)
    fin_character = img_float[0] > 0.5
    for i in range(max_depth):
        row = img_float[i]
        is_finished = row > 0.5
        
    
    
    # Calculate average intensity per column (inverted: dark = high)
    inverted = 1.0 - img_float
    column_intensities = np.mean(inverted, axis=0)
    column_intensities = column_intensities 

    # Threshold for visualization
    threshold = 0.1
    content_cols_normal = np.sum(column_intensities < (1.0 - threshold)) # Dark content
    

    # Compute mean intensity across all columns
    mean_intensity = np.mean(column_intensities) / 2
    
    

    # Plot results (only if save_plot is True)
    if save_plot:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.plot(column_intensities, label='Column Intensities', color='blue')
        plt.title('Column Intensities (Normal)')
        plt.xlabel('Column')
        plt.ylabel('Average Intensity')

        # Add threshold line
        plt.axhline(y=1.0 - threshold, color='r', linestyle='--', label=f'Threshold {1.0 - threshold}')

        # Add mean line (dark yellow)
        plt.axhline(y=mean_intensity, color='#b58900', linestyle='-', linewidth=2, label=f'Mean = {mean_intensity:.3f}')

        plt.legend()
        plt.tight_layout()

        output_path = f"results/inspec_{number}.png"
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        plt.close()

    return column_intensities

def aspect_ratio_frequency(img):
    """Частота соотношения сторон."""
    # i need to calcula
    column_intensities = calculate_column_intensity(img)
    threshold = 0.05
    col = np.asarray(column_intensities).astype(float).ravel()
    n = col.size
    if n == 0:
        return None
    
     # Normalize to [0,1] if values appear to be in a larger scale (e.g., 0..255)
    if col.max() > 1.0:
        col = col / float(col.max())
    states = col > threshold
    
    # calculate amount of false in states
    false_count = np.sum(states == False)
    true_count = np.sum(states == True)
    # print(f"False count: {false_count}, True count: {true_count}")
    
    ratio = false_count / img.shape[1]
    run_start = 0
    in_character = False
    peaks = []
    #calculate the lenth horizontally of each peak and divide by the peaks count
    for i in range(len(states)):
        if states[i] == True:
            if not in_character:
                run_start = i
                in_character = True
        else:
            if in_character:
                run_end = i
                run_width = run_end - run_start
                if run_width >= 2:
                    peaks.append((run_end- run_start + 2))
                    # print(f"run_width: {run_width}")
                    
                in_character = False
    amount_of_peaks = len(peaks)
    if amount_of_peaks > 0:
        average_peak_width = sum(peaks) / amount_of_peaks
    else:
        average_peak_width = 0
            
            
    # return average_peak_width as between 0 and 1
    average_peak_width = average_peak_width / img.shape[1]
    return ratio, average_peak_width   


class FeatureExtractor:
    """Извлечение признаков из изображения ячейки."""
    
    @staticmethod
    def extract_features(img: np.ndarray) -> Dict[str, float]:
        """
        Извлечь все признаки из изображения.
        
        Args:
            img: Нормализованное изображение [0, 1]
        
        Returns:
            Словарь с признаками
        """
        # Конвертация в uint8 для OpenCV
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Инвертируем для работы (чёрный текст на белом -> белый текст на чёрном)
        img_inv = 255 - img_uint8
        
        # 1. dark_frac — доля чёрных пикселей
        dark_frac = np.sum(img < 0.5) / img.size
        
        # 2 & 3. Вертикальные и горизонтальные штрихи
        vert_strokes = FeatureExtractor._count_vertical_strokes(img_uint8)
        horiz_strokes = FeatureExtractor._count_horizontal_strokes(img_uint8)
        
        # 4. Aspect ratio
        aspect_ratio = FeatureExtractor._calculate_aspect_ratio(img_inv)
        
        # 5. Compactness
        compactness = FeatureExtractor._calculate_compactness(img_inv)
        
        # 6. Number of contours
        num_contours = FeatureExtractor._count_contours(img_uint8)
        
        # 7. Connected components count
        cc_count = FeatureExtractor._count_connected_components(img_inv)
        
        # 8 & 9. Projection peaks
        col_peaks = FeatureExtractor._count_column_peaks(img_uint8)
        row_peaks = FeatureExtractor._count_row_peaks(img_uint8)
        
        col_profile = FeatureExtractor._column_profile(img_uint8)
        vertical_intensity_variance = FeatureExtractor._vertical_intensity_variance(img_uint8)
        gap_depth_index = FeatureExtractor._gap_depth_index(img_uint8)
        intensity_fluctuation_ratio = FeatureExtractor._intensity_fluctuation_ratio(img_uint8)
        column_intensity_entropy=FeatureExtractor._column_intensity_entropy(img_uint8)
        
        # aspect_ratio_frequency(img_uint8)
        frequency_white, average_peak_width = aspect_ratio_frequency(img_uint8)

        
        return {
            'dark_frac': dark_frac,
            'vert_strokes_count': vert_strokes,
            'horiz_strokes_count': horiz_strokes,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'num_contours': num_contours,
            'cc_count': cc_count,
            'col_peaks': col_peaks,
            'row_peaks': row_peaks,
            'vertical_intensity_variance': vertical_intensity_variance,
            'gap_depth_index': gap_depth_index,
            "intensity_fluctuation_ratio": intensity_fluctuation_ratio,
            "column_intensity_entropy": column_intensity_entropy,
            "frequency_white": frequency_white,
            "average_peak_width": average_peak_width,
            
            
            
        }
    
    @staticmethod
    def _count_vertical_strokes(img: np.ndarray) -> int:
        """Подсчёт вертикальных штрихов через морфологию."""
        # Вертикальное ядро
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Бинаризация
        _, binary = cv2.threshold(vertical, 127, 255, cv2.THRESH_BINARY)
        
        # Подсчёт связных компонент
        num_labels, _ = cv2.connectedComponents(binary)
        return max(0, num_labels - 1)  # -1 для фона
    
    @staticmethod
    def _count_horizontal_strokes(img: np.ndarray) -> int:
        """Подсчёт горизонтальных штрихов через морфологию."""
        # Горизонтальное ядро
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Бинаризация
        _, binary = cv2.threshold(horizontal, 127, 255, cv2.THRESH_BINARY)
        
        # Подсчёт связных компонент
        num_labels, _ = cv2.connectedComponents(binary)
        return max(0, num_labels - 1)
    
    @staticmethod
    def _calculate_aspect_ratio(img: np.ndarray) -> float:
        """Вычисление соотношения сторон содержимого."""
        # Найти bounding box содержимого
        coords = cv2.findNonZero(img)
        if coords is None or len(coords) < 2:
            return 1.0
        
        x, y, w, h = cv2.boundingRect(coords)
        if h == 0:
            return 1.0
        # return value between 0 and 1 
        aspect_ratio = w / h
        aspect_ratio = aspect_ratio / np.max([w, h])
        return aspect_ratio
    
    @staticmethod
    def _calculate_compactness(img: np.ndarray) -> float:
        """Компактность: отношение площади содержимого к площади bounding box."""
        coords = cv2.findNonZero(img)
        if coords is None or len(coords) < 2:
            return 0.0
        
        x, y, w, h = cv2.boundingRect(coords)
        bbox_area = w * h
        if bbox_area == 0:
            return 0.0
        
        content_area = np.sum(img > 0)
        return content_area / bbox_area
    
    @staticmethod
    def _count_contours(img: np.ndarray) -> int:
        """Подсчёт контуров."""
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Фильтруем мелкие контуры (шум)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 5]
        return len(significant_contours)
    
    @staticmethod
    def _count_connected_components(img: np.ndarray) -> int:
        """Подсчёт связных компонент."""
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(binary)
        return max(0, num_labels - 1)  # -1 для фона
    
    @staticmethod
    def _count_column_peaks(img: np.ndarray) -> int:
        """Подсчёт пиков в вертикальной проекции (столбцы)."""
        # Вертикальная проекция (сумма по столбцам)
        projection = np.sum(img, axis=0)
        
        if np.max(projection) < 10:  # Слишком мало содержимого
            return 0
        
        # Нормализация
        if np.max(projection) > 0:
            projection = projection / np.max(projection)
        
        # Поиск пиков
        peaks, _ = find_peaks(projection, height=0.2, distance=3)
        return len(peaks)
    
    @staticmethod
    def _count_row_peaks(img: np.ndarray) -> int:
        """Подсчёт пиков в горизонтальной проекции (строки)."""
        # Горизонтальная проекция (сумма по строкам)
        projection = np.sum(img, axis=1)
        
        if np.max(projection) < 10:
            return 0
        
        # Нормализация
        if np.max(projection) > 0:
            projection = projection / np.max(projection)
        
        # Поиск пиков
        peaks, _ = find_peaks(projection, height=0.2, distance=2)
        return len(peaks)
    
    @staticmethod
    def _column_profile(img: np.ndarray) -> np.ndarray:
        """
        Compute inverted column intensity profile:
         - convert to grayscale float [0,1]
         - invert so that ink/dark -> larger values (1.0 = darkest)
         - return 1D array of length = image width
        """
        gray = img
        # column mean (white ~1.0) -> invert to make ink high
        col_mean = gray.mean(axis=0)
        col_profile = 1.0 - col_mean
        # small numerical stabilizer
        return col_profile.astype(np.float32)

    @staticmethod
    def _vertical_intensity_variance(img: np.ndarray) -> float:
        """
        Vertical intensity variance:
        variance of the column profile (higher -> more variability between columns).
        Возвращает float (ненормализованное значение дисперсии, >=0).
        """
        col = FeatureExtractor._column_profile(img)
        if col.size == 0:
            return 0.0
        return float(np.var(col))

    @staticmethod
    def _gap_depth_index(img: np.ndarray) -> float:
        """
        Gap Depth Index:
        Средняя глубина провалов (valleys) между соседними пиками столбцовой проекции,
        нормализованная по полному динамическому диапазону профиля.
        Алгоритм:
          - находим локальные пики и локальные минимумы (сравнение с соседями)
          - для каждой впадины между двумя ближайшими пиками считаем depth = min(peak_left, peak_right) - valley
          - возвращаем среднюю depth, нормализованную на (max-min) диапазон
        Если пиков/впадин мало, используется fallback = (global_max - global_min)
        Возвращаем 0..1 (0 = нет провалов, 1 = максимально глубокие провалы).
        """
        col = FeatureExtractor._column_profile(img)
        n = col.size
        if n < 3:
            return 0.0

        # normalize to 0..1 for robust numeric behavior
        minv = float(col.min())
        maxv = float(col.max())
        rng = maxv - minv
        if rng <= 1e-8:
            return 0.0
        norm = (col - minv) / rng

        # local maxima/minima detection (strict)
        left = norm[:-2]
        center = norm[1:-1]
        right = norm[2:]
        maxima_idx = np.where((center > left) & (center >= right))[0] + 1
        minima_idx = np.where((center < left) & (center <= right))[0] + 1

        # If there are not at least two peaks, fallback to global contrast
        if maxima_idx.size < 2 or minima_idx.size == 0:
            # fallback: normalized global peak-to-valley
            return float(1.0 if rng > 0 else 0.0) * float((maxv - minv) / (maxv + 1e-9))

        # For each valley, find nearest peaks to left and right (if exist) and compute depth
        depths = []
        for valley in minima_idx:
            left_peaks = maxima_idx[maxima_idx < valley]
            right_peaks = maxima_idx[maxima_idx > valley]
            if left_peaks.size == 0 or right_peaks.size == 0:
                continue
            left_peak = left_peaks[-1]
            right_peak = right_peaks[0]
            peak_height = min(norm[left_peak], norm[right_peak])
            valley_val = norm[valley]
            depth = peak_height - valley_val
            if depth > 0:
                depths.append(depth)

        if len(depths) == 0:
            # fallback to difference between neighbor peaks and global min
            return float((norm.max() - norm.min()))

        # average and clamp to [0,1]
        avg_depth = float(np.mean(depths))
        return float(np.clip(avg_depth, 0.0, 1.0))

    @staticmethod
    def _intensity_fluctuation_ratio(img: np.ndarray) -> float:
        """
        Intensity fluctuation ratio:
        Средняя абсолютная разница между смежными столбцами, нормализованная на среднюю интенсивность.
        Формула: mean(|diffs|) / (mean(col_profile) + eps)
        Чем выше — тем более "рваный" профиль (частые резкие перепады).
        Возвращает неотрицательное float.
        """
        col = FeatureExtractor._column_profile(img)
        if col.size < 2:
            return 0.0
        diffs = np.abs(np.diff(col))
        mean_abs_diff = float(diffs.mean())
        mean_col = float(np.abs(col).mean())
        eps = 1e-9
        ratio = mean_abs_diff / (mean_col + eps)
        return float(ratio)

    @staticmethod
    def _column_intensity_entropy(img: np.ndarray, bins: int = 32) -> float:
        """
        Column intensity entropy:
        Энтропия распределения значений столцового профиля.
        Рассчитывается как Shannon entropy на гистограмме column_profile (bins bins).
        Нормализуется делением на log2(bins) для значений в диапазоне [0,1].
        """
        col = FeatureExtractor._column_profile(img)
        if col.size == 0:
            return 0.0
        # histogram on [0,1]
        counts, _ = np.histogram(col, bins=bins, range=(0.0, 1.0))
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts.astype(np.float64) / float(total)
        p_nonzero = p[p > 0.0]
        entropy = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
        # normalize to [0,1] by dividing by log2(bins)
        max_ent = np.log2(bins)
        if max_ent <= 0:
            return 0.0
        return float(entropy / max_ent)


def generate_dataset(n_samples_per_class: int = 100, 
                     img_size: Tuple[int, int] = (160,40)) -> pd.DataFrame:
    """
    Генерация полного датасета с признаками.
    
    Args:
        n_samples_per_class: Количество образцов на класс
        img_size: Размер изображения (width, height)
    
    Returns:
        DataFrame с признаками и метками
    """
    print(f"Генерация датасета: {n_samples_per_class} образцов на класс")
    print(f"Размер изображений: {img_size}")
    print("="*60)
    
    generator = CellImageGenerator(img_size)
    extractor = FeatureExtractor()
    
    data = []
    images = []
    count = 0
    # Генерация чисел
    print("Генерация NUMBERS...")
    for i in range(n_samples_per_class):
        img = generator.generate_number()
        features = extractor.extract_features(img)
        if count <= 5:
            cv2.imwrite(f"results/number_{i}.png", (img * 255).astype(np.uint8))
            count += 1
        features['label'] = 'NUMBER'
        features['sample_id'] = f'NUM_{i:03d}'
        data.append(features)
        images.append(('NUMBER', img))
    
    # Генерация слов
    print("Генерация WORDS...")
    for i in range(n_samples_per_class):
        img = generator.generate_word()
        features = extractor.extract_features(img)
        features['label'] = 'WORD'
        features['sample_id'] = f'WORD_{i:03d}'
        if count <= 10:
            cv2.imwrite(f"results/word_{i}.png", (img * 255).astype(np.uint8))
            count += 1
        data.append(features)
        images.append(('WORD', img))
    
    # Генерация других
    print("Генерация OTHER...")
    for i in range(n_samples_per_class):
        img = generator.generate_other()
        features = extractor.extract_features(img)
        if count <= 15:
            cv2.imwrite(f"results/other_{i}.png", (img * 255).astype(np.uint8))
            count += 1
        features['label'] = 'OTHER'
        features['sample_id'] = f'OTHER_{i:03d}'
        data.append(features)
        images.append(('OTHER', img))
    
    df = pd.DataFrame(data)
    
    # Переупорядочить столбцы
    cols = ['sample_id', 'label'] + [c for c in df.columns if c not in ['sample_id', 'label']]
    df = df[cols]
    
    print("\nДатасет сгенерирован!")
    print(f"Всего образцов: {len(df)}")
    print(f"\nРаспределение классов:")
    print(df['label'].value_counts())
    
    return df, images


def visualize_samples(images: List[Tuple[str, np.ndarray]], 
                     n_per_class: int = 5,
                     save_path: str = 'results/sample_images.png'):
    """Визуализация примеров изображений."""
    classes = ['NUMBER', 'WORD', 'OTHER']
    
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(15, 9))
    
    for i, class_name in enumerate(classes):
        class_images = [img for label, img in images if label == class_name]
        
        for j in range(n_per_class):
            ax = axes[i, j]
            
            if j < len(class_images):
                ax.imshow(class_images[j], cmap='gray', vmin=0, vmax=1)
                if j == 0:
                    ax.set_ylabel(class_name, fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No sample', ha='center', va='center')
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nПримеры изображений сохранены в: {save_path}")
    plt.close()


def visualize_feature_distributions(df: pd.DataFrame, 
                                    save_path: str = 'results/feature_distributions.png'):
    """Визуализация распределений признаков по классам."""
    features = [col for col in df.columns if col not in ['sample_id', 'label']]
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        for label in ['NUMBER', 'WORD', 'OTHER']:
            data = df[df['label'] == label][feature]
            ax.hist(data, alpha=0.5, label=label, bins=20)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Частота')
        ax.set_title(f'Распределение: {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Скрыть лишние оси
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Распределения признаков сохранены в: {save_path}")
    plt.close()

def visualize_feature_pairs(df: pd.DataFrame, 
                            feature_pairs: List[Tuple[str, str]] = None,
                            save_path: str = 'results/feature_pairs.png'):
    """Визуализация пар признаков для анализа разделимости классов."""
    if feature_pairs is None:
        # Самые информативные пары
        feature_pairs = [
            ('dark_frac', 'col_peaks'),
            # ('vert_strokes_count', 'aspect_ratio'),
            # ('compactness', 'cc_count'),
            # ('num_contours', 'col_peaks'),
            # ("vert_strokes_count", "compactness"),
            ("cc_count", "col_peaks"),
            # ("num_contours", "vert_strokes_count"),
            ('dark_frac', 'col_peaks'),
            ('aspect_ratio', 'compactness'),
            ("dark_frac", "intensity_fluctuation_ratio"),
            ("dark_frac", "vertical_intensity_variance"),
            # ('vert_strokes_count', 'aspect_ratio'),
            # ('compactness', 'cc_count'),
            # ("aspect_ratio", "vertical_intensity_variance"),
            # ("compactness", "vertical_intensity_variance"),
            # ('num_contours', 'col_peaks'),
            # ("vert_strokes_count", "compactness"),
            # ("cc_count", "col_peaks"),
            # ("col_peaks", "row_peaks"),
            # ("col_peaks", "vertical_intensity_variance"),
            # ("num_contours", "vert_strokes_count"),
            # ("intensity_fluctuation_ratio", "dark_frac"),
            # ("intensity_fluctuation_ratio", "column_intensity_entropy"),
            ("intensity_fluctuation_ratio", "compactness"),
            ("col_peaks", "column_intensity_entropy"),
            ("gap_depth_index", "col_peaks"),
            ("vertical_intensity_variance", "compactness"),
            ("frequency_white", "compactness"),
            ("frequency_white", "aspect_ratio"),
            ("frequency_white", "vertical_intensity_variance"),
            ("frequency_white", "average_peak_width"),
            ("average_peak_width", "compactness"),
            ("average_peak_width", "aspect_ratio"),
            ("average_peak_width", "vertical_intensity_variance"),
            ("average_peak_width", "intensity_fluctuation_ratio"),
            ("col_peaks", "average_peak_width"),
            ("col_peaks", "frequency_white"),
            ("col_peaks", "intensity_fluctuation_ratio"),
            
            ("col_peaks", "vertical_intensity_variance"),
            # ("column_intensity_entropy", "gap_depth_index"),
            # ("gap_depth_index", "vertical_intensity_variance"),
        ]
    
    n_pairs = len(feature_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    
    colors = {'NUMBER': 'red', 'WORD': 'blue', 'OTHER': 'green'}
    
    for i, (feat1, feat2) in enumerate(feature_pairs):
        ax = axes[i]
        
        for label in ['NUMBER', 'WORD', 'OTHER']:
            data = df[df['label'] == label]
            ax.scatter(data[feat1], data[feat2], 
                      alpha=0.6, label=label, c=colors[label], s=30)
        
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_title(f'{feat1} vs {feat2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Скрыть лишние оси
    for i in range(n_pairs, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Пары признаков сохранены в: {save_path}")
    plt.close()


def print_feature_statistics(df: pd.DataFrame):
    """Вывод статистики признаков по классам."""
    print("\n" + "="*60)
    print("СТАТИСТИКА ПРИЗНАКОВ ПО КЛАССАМ")
    print("="*60)
    
    features = [col for col in df.columns if col not in ['sample_id', 'label']]
    
    for feature in features:
        print(f"\n{feature.upper()}:")
        print("-" * 50)
        stats = df.groupby('label')[feature].agg(['mean', 'std', 'min', 'max'])
        print(stats.to_string())


def explain_features():
    """Объяснение признаков и их значимости."""
    print("\n" + "="*60)
    print("ОБЪЯСНЕНИЕ ПРИЗНАКОВ")
    print("="*60)
    
    explanations = {
        'dark_frac': '''
Доля чёрных пикселей (после нормализации)
- NUMBER: Обычно высокая плотность для цифр с жирным шрифтом
- WORD: Средняя плотность, зависит от длины слова
- OTHER: Низкая плотность для пустых ячеек, высокая для линий
        ''',
        
        'vert_strokes_count': '''
Число вертикальных штрихов (морфология)
- NUMBER: Цифры 1, 4, 7 дают много вертикальных штрихов
- WORD: Буквы также имеют вертикали, но менее регулярные
- OTHER: Мало вертикалей, кроме специфичных символов
        ''',
        
        'horiz_strokes_count': '''
Число горизонтальных штрихов (морфология)
- NUMBER: Цифры типа 5, 7, знак минус
- WORD: Буквы с горизонталями (Е, Н, Т)
- OTHER: Табличные линии дают много горизонтальных штрихов
        ''',
        
        'aspect_ratio': '''
Соотношение ширина/высота содержимого
- NUMBER: Компактные, AR близко к 1-3 для коротких чисел
- WORD: Слова шире, AR > 3-5
- OTHER: Зависит от типа (линии дают экстремальные значения)
        ''',
        
        'compactness': '''
Заполненность bounding box (area_nonzero / bbox_area)
- NUMBER: Высокая компактность для цифр
- WORD: Ниже, т.к. между буквами есть пробелы
- OTHER: Очень низкая для пустых, высокая для одного символа
        ''',
        
        'num_contours': '''
Количество контуров (≈ число символов после бинаризации)
- NUMBER: 1-5 контуров для чисел с разделителями
- WORD: Больше контуров для длинных слов (5-15)
- OTHER: 0-2 для пустых или одиночных символов
        ''',
        
        'cc_count': '''
Количество связных компонент
- NUMBER: Похоже на num_contours, но строже
- WORD: Больше компонент для текста
- OTHER: Мало компонент
        ''',
        
        'col_peaks': '''
Число пиков в вертикальной проекции (≈ счётчик символов)
- NUMBER: 1-5 пиков для цифр
- WORD: 5-15 пиков для слов
- OTHER: 0-2 пика
        ''',
        
        'row_peaks': '''
Число пиков в горизонтальной проекции
- Вспомогательный признак для многострочного содержимого
- Обычно 1-2 пика для одной строки
        '''
    }
    
    for feature, explanation in explanations.items():
        print(f"\n{feature}:")
        print(explanation.strip())


def main():
    """Основная функция."""
    print("="*60)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА И ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ")
    print("="*60)
    
    # Генерация датасета
    df, images = generate_dataset(n_samples_per_class=300, img_size=(200, 32))
    
    # Сохранение в CSV
    csv_path = 'results/cell_features_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nДатасет сохранён в: {csv_path}")
    
    # Статистика
    print_feature_statistics(df)
    
    
    # Объяснение признаков
    explain_features()
    
    # Визуализация
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ")
    print("="*60)
    
    visualize_samples(images, n_per_class=5)
    visualize_feature_distributions(df)
    visualize_feature_pairs(df)
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print(f"\nФайлы:")
    print(f"  - {csv_path}")
    print(f"  - sample_images.png")
    print(f"  - feature_distributions.png")
    print(f"  - feature_pairs.png")
    
    return df, images


if __name__ == '__main__':
    df, images = main()