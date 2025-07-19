#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Debug Test Script
Тестовый скрипт для отладки OCR распознавания изображений
"""

import os
import sys
from pathlib import Path

# OCR библиотеки
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    print("✅ Все OCR библиотеки загружены успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)


def check_tesseract_installation():
    """Проверка установки Tesseract"""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract версия: {version}")
        
        languages = pytesseract.get_languages()
        print(f"✅ Доступные языки: {languages}")
        
        if 'eng' in languages:
            print("✅ Английский язык поддерживается")
        else:
            print("⚠️ Английский язык не найден")
            
        return True
    except Exception as e:
        print(f"❌ Ошибка Tesseract: {e}")
        return False


def simple_ocr_test(image_path):
    """Простой тест OCR без дополнительной обработки"""
    try:
        print(f"\n=== ПРОСТОЙ OCR ТЕСТ ===")
        print(f"Файл: {image_path}")
        
        # Проверка файла
        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}")
            return None
            
        # Открытие изображения
        image = Image.open(image_path)
        print(f"✅ Изображение открыто: {image.size}, режим: {image.mode}")
        
        # Простое извлечение текста
        text = pytesseract.image_to_string(image, lang='eng')
        print(f"📝 Извлеченный текст ({len(text)} символов):")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)
        
        return text
        
    except Exception as e:
        print(f"❌ Ошибка простого OCR: {e}")
        return None


def advanced_ocr_test(image_path):
    """Продвинутый тест OCR с предобработкой"""
    try:
        print(f"\n=== ПРОДВИНУТЫЙ OCR ТЕСТ ===")
        print(f"Файл: {image_path}")
        
        # Открытие изображения
        image = Image.open(image_path)
        print(f"✅ Исходное изображение: {image.size}")
        
        # Конвертация в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"✅ Конвертировано в RGB")
        
        # Масштабирование маленьких изображений
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000/width, 1000/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"✅ Масштабировано до: {new_width}x{new_height}")
        
        # Улучшение контраста
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        print(f"✅ Контраст улучшен")
        
        # Улучшение резкости
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        print(f"✅ Резкость улучшена")
        
        # Конвертация в оттенки серого
        image = image.convert('L')
        print(f"✅ Конвертировано в оттенки серого")
        
        # Применение фильтра
        image = image.filter(ImageFilter.MedianFilter(size=3))
        print(f"✅ Фильтр применен")
        
        # БЕЗОПАСНАЯ конфигурация OCR (без проблемных символов)
        safe_config = r'--oem 3 --psm 6'
        
        print(f"✅ Используется конфигурация: {safe_config}")
        
        # Извлечение текста
        text = pytesseract.image_to_string(image, lang='eng', config=safe_config)
        
        print(f"📝 Извлеченный текст ({len(text)} символов):")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)
        
        # Анализ качества
        if text.strip():
            letters = sum(c.isalpha() for c in text)
            total_chars = len(text.replace(' ', '').replace('\n', ''))
            quality_score = letters / total_chars if total_chars > 0 else 0
            
            print(f"📊 Анализ качества:")
            print(f"   Буквы: {letters}")
            print(f"   Всего символов: {total_chars}")
            print(f"   Качество: {quality_score:.2f}")
            
            if quality_score > 0.3:
                print(f"✅ Качество хорошее")
            else:
                print(f"⚠️ Качество низкое")
        else:
            print(f"⚠️ Текст не извлечен")
        
        return text
        
    except Exception as e:
        print(f"❌ Ошибка продвинутого OCR: {e}")
        return None


def test_opencv_processing(image_path):
    """Тест обработки с OpenCV"""
    try:
        print(f"\n=== OPENCV ОБРАБОТКА ===")
        
        # Чтение изображения с OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ OpenCV не может прочитать файл")
            return None
            
        print(f"✅ OpenCV прочитал изображение: {img.shape}")
        
        # Конвертация в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"✅ Конвертировано в серый")
        
        # Применение размытия для удаления шума
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        print(f"✅ Размытие применено")
        
        # Бинаризация
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"✅ Бинаризация выполнена")
        
        # Конвертация обратно в PIL для OCR
        pil_image = Image.fromarray(thresh)
        
        # OCR с обработанным изображением
        text = pytesseract.image_to_string(pil_image, lang='eng', config=r'--oem 3 --psm 6')
        
        print(f"📝 Текст после OpenCV обработки ({len(text)} символов):")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)
        
        return text
        
    except Exception as e:
        print(f"❌ Ошибка OpenCV обработки: {e}")
        return None


def test_different_configs(image_path):
    """Тест разных конфигураций OCR"""
    try:
        print(f"\n=== ТЕСТ РАЗНЫХ КОНФИГУРАЦИЙ ===")
        
        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')
        
        configs = [
            ('Базовая', ''),
            ('OEM 3 PSM 6', '--oem 3 --psm 6'),
            ('OEM 3 PSM 7', '--oem 3 --psm 7'),
            ('OEM 3 PSM 8', '--oem 3 --psm 8'),
            ('OEM 1 PSM 6', '--oem 1 --psm 6'),
        ]
        
        results = []
        
        for name, config in configs:
            try:
                print(f"\n🔧 Тестируем: {name} ({config})")
                text = pytesseract.image_to_string(image, lang='eng', config=config)
                
                if text.strip():
                    letters = sum(c.isalpha() for c in text)
                    total = len(text.replace(' ', '').replace('\n', ''))
                    quality = letters / total if total > 0 else 0
                    
                    print(f"   Длина: {len(text)}, Качество: {quality:.2f}")
                    print(f"   Превью: {text[:100]}...")
                    
                    results.append((name, len(text), quality, text))
                else:
                    print(f"   ❌ Текст не извлечен")
                    results.append((name, 0, 0, ""))
                    
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                results.append((name, 0, 0, ""))
        
        # Лучший результат
        if results:
            best = max(results, key=lambda x: x[2])  # По качеству
            print(f"\n🏆 ЛУЧШИЙ РЕЗУЛЬТАТ: {best[0]}")
            print(f"   Качество: {best[2]:.2f}")
            print(f"   Длина: {best[1]}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка тестирования конфигураций: {e}")
        return []


def find_test_images(directory="./data/634/2025"):
    """Поиск тестовых изображений"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    images = []
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    images.append(os.path.join(root, file))
                    if len(images) >= 5:  # Максимум 5 изображений для теста
                        break
            if len(images) >= 5:
                break
                
        return images
    except Exception as e:
        print(f"❌ Ошибка поиска изображений: {e}")
        return []


def main():
    """Главная функция тестирования"""
    print("🔍 OCR DEBUG TEST SCRIPT")
    print("=" * 50)
    
    # Проверка Tesseract
    if not check_tesseract_installation():
        return
    
    # Поиск тестовых изображений
    print(f"\n🔍 Поиск тестовых изображений...")
    test_images = find_test_images()
    
    if not test_images:
        print("❌ Тестовые изображения не найдены")
        
        # Ручной ввод пути
        manual_path = input("\n📁 Введите путь к изображению для теста (или Enter для выхода): ").strip()
        if manual_path and os.path.exists(manual_path):
            test_images = [manual_path]
        else:
            print("❌ Выход из программы")
            return
    
    print(f"✅ Найдено {len(test_images)} изображений для теста")
    
    # Тестирование первого изображения
    test_image = test_images[0]
    print(f"\n🖼️ ТЕСТИРУЕМ: {os.path.basename(test_image)}")
    print("=" * 50)
    
    # Запуск всех тестов
    simple_ocr_test(test_image)
    advanced_ocr_test(test_image)
    test_opencv_processing(test_image)
    test_different_configs(test_image)
    
    print(f"\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 50)
    
    # Показать все найденные изображения
    if len(test_images) > 1:
        print(f"\n📋 Другие найденные изображения:")
        for i, img in enumerate(test_images[1:], 2):
            print(f"   {i}. {os.path.basename(img)}")
        
        test_more = input(f"\n❓ Протестировать другое изображение? (введите номер или Enter для выхода): ").strip()
        if test_more.isdigit():
            idx = int(test_more) - 1
            if 0 <= idx < len(test_images):
                main_test_image = test_images[idx]
                print(f"\n🖼️ ТЕСТИРУЕМ: {os.path.basename(main_test_image)}")
                simple_ocr_test(main_test_image)
                advanced_ocr_test(main_test_image)


if __name__ == "__main__":
    main()
