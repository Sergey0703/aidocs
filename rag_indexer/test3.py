#!/usr/bin/env python3
"""
Тест для проверки работы улучшенного PDF процессора
"""

import os
import sys
from pathlib import Path

# Добавить текущую директорию к пути
sys.path.append('.')

def test_pdf_libraries():
    """Проверка доступности библиотек PDF"""
    print("🔍 Проверка библиотек PDF:")
    
    libraries = {
        'PyMuPDF': False,
        'pdfplumber': False,
        'pdf2image': False
    }
    
    try:
        import fitz
        libraries['PyMuPDF'] = True
        print("  ✅ PyMuPDF доступен")
    except ImportError:
        print("  ❌ PyMuPDF НЕ доступен - установите: pip install PyMuPDF")
    
    try:
        import pdfplumber
        libraries['pdfplumber'] = True
        print("  ✅ pdfplumber доступен")
    except ImportError:
        print("  ❌ pdfplumber НЕ доступен - установите: pip install pdfplumber")
    
    try:
        from pdf2image import convert_from_path
        libraries['pdf2image'] = True
        print("  ✅ pdf2image доступен")
    except ImportError:
        print("  ❌ pdf2image НЕ доступен - установите: pip install pdf2image")
    
    return libraries

def test_config_loading():
    """Проверка загрузки конфигурации PDF"""
    print("\n📋 Проверка конфигурации PDF:")
    
    try:
        from config import get_config
        config = get_config()
        
        # Проверка PDF настроек
        pdf_enabled = config.is_feature_enabled('enhanced_pdf_processing')
        print(f"  Enhanced PDF processing: {'✅ ENABLED' if pdf_enabled else '❌ DISABLED'}")
        
        if pdf_enabled:
            pdf_settings = config.get_pdf_processing_settings()
            print(f"  PDF chunk size: {pdf_settings['chunk_size']}")
            print(f"  Auto method selection: {'✅' if pdf_settings['auto_method_selection'] else '❌'}")
            print(f"  Table extraction: {'✅' if pdf_settings['enable_table_extraction'] else '❌'}")
            print(f"  OCR fallback: {'✅' if pdf_settings['enable_ocr_fallback'] else '❌'}")
        
        return config
    except Exception as e:
        print(f"  ❌ Ошибка загрузки конфигурации: {e}")
        return None

def test_pdf_processor():
    """Проверка создания PDF процессора"""
    print("\n🔧 Проверка создания PDF процессора:")
    
    try:
        from document_parsers import EnhancedPDFProcessor
        from config import get_config
        
        config = get_config()
        pdf_processor = EnhancedPDFProcessor(config)
        
        print("  ✅ EnhancedPDFProcessor создан успешно")
        
        # Проверка доступности библиотек в процессоре
        libs = pdf_processor.libraries_available
        print(f"  PyMuPDF в процессоре: {'✅' if libs['pymupdf'] else '❌'}")
        print(f"  pdfplumber в процессоре: {'✅' if libs['pdfplumber'] else '❌'}")
        print(f"  pdf2image в процессоре: {'✅' if libs['pdf2image'] else '❌'}")
        
        return pdf_processor
    except Exception as e:
        print(f"  ❌ Ошибка создания PDF процессора: {e}")
        return None

def find_test_pdf():
    """Найти тестовый PDF файл"""
    print("\n📄 Поиск тестового PDF:")
    
    # Проверить директории из конфигурации
    test_dirs = [
        "./data/634/2025/1",
        "./data/634/2025",
        "./data",
        "."
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        print(f"  ✅ Найден PDF: {pdf_path}")
                        return pdf_path
    
    print("  ❌ PDF файлы не найдены")
    return None

def test_pdf_processing(pdf_processor, pdf_file):
    """Тест обработки PDF файла"""
    print(f"\n🧪 Тест обработки PDF: {os.path.basename(pdf_file)}")
    
    try:
        # Анализ PDF
        pdf_analysis = pdf_processor.detect_pdf_type(pdf_file)
        print(f"  PDF тип: {pdf_analysis['type']}")
        print(f"  Страниц: {pdf_analysis['page_count']}")
        print(f"  Рекомендуемый метод: {pdf_analysis['recommended_method']}")
        print(f"  Текстовое покрытие: {pdf_analysis['text_coverage']:.2f}")
        
        # Обработка PDF
        documents = pdf_processor.process_pdf_file(pdf_file)
        
        if documents:
            doc = documents[0]
            content_length = len(doc.text)
            extraction_method = doc.metadata.get('extraction_info', {}).get('method', 'unknown')
            
            print(f"  ✅ Успешно обработан!")
            print(f"  Извлечено символов: {content_length:,}")
            print(f"  Метод извлечения: {extraction_method}")
            
            # Показать превью содержимого
            preview = doc.text[:200].replace('\n', ' ')
            print(f"  Превью: \"{preview}...\"")
            
            return True
        else:
            print("  ❌ Не удалось извлечь содержимое")
            return False
            
    except Exception as e:
        print(f"  ❌ Ошибка обработки: {e}")
        return False

def main():
    """Основная функция теста"""
    print("🔧 ТЕСТ УЛУЧШЕННОГО PDF ПРОЦЕССОРА")
    print("=" * 50)
    
    # 1. Проверка библиотек
    libraries = test_pdf_libraries()
    
    # 2. Проверка конфигурации
    config = test_config_loading()
    
    # 3. Проверка создания процессора
    pdf_processor = test_pdf_processor()
    
    # 4. Поиск тестового PDF
    pdf_file = find_test_pdf()
    
    # 5. Тест обработки (если все доступно)
    if pdf_processor and pdf_file:
        success = test_pdf_processing(pdf_processor, pdf_file)
        
        print(f"\n🎯 РЕЗУЛЬТАТ ТЕСТА:")
        if success:
            print("  ✅ Улучшенный PDF процессор работает КОРРЕКТНО!")
        else:
            print("  ❌ Улучшенный PDF процессор НЕ работает должным образом")
    else:
        print(f"\n❌ ТЕСТ НЕ ЗАВЕРШЕН:")
        if not pdf_processor:
            print("  - PDF процессор не создан")
        if not pdf_file:
            print("  - Тестовый PDF файл не найден")
    
    print("\n💡 РЕКОМЕНДАЦИИ:")
    if not all(libraries.values()):
        print("  - Установите недостающие библиотеки PDF")
    if not config or not config.is_feature_enabled('enhanced_pdf_processing'):
        print("  - Проверьте настройки PDF в .env файле")
    if not pdf_file:
        print("  - Добавьте PDF файлы в директорию для тестирования")

if __name__ == "__main__":
    main()
