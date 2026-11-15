"""
Language Switcher Script for Graph RAG Studio.

This script allows easy switching between Arabic and English modes
by enabling/disabling the .env.arabic file.

Usage:
    python scripts/switch_language.py arabic    # Enable Arabic mode
    python scripts/switch_language.py english   # Enable English mode
    python scripts/switch_language.py status    # Show current mode
"""

import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARABIC_ENV_FILE = PROJECT_ROOT / ".env.arabic"
ARABIC_ENV_DISABLED = PROJECT_ROOT / ".env.arabic.disabled"


def enable_arabic() -> None:
    """Enable Arabic mode by ensuring .env.arabic exists."""
    if ARABIC_ENV_DISABLED.exists():
        ARABIC_ENV_DISABLED.rename(ARABIC_ENV_FILE)
        print("✅ تم تفعيل الوضع العربي (Arabic mode enabled)")
    elif ARABIC_ENV_FILE.exists():
        print("ℹ️  الوضع العربي مفعل بالفعل (Arabic mode already enabled)")
    else:
        # Create the file if it doesn't exist
        ARABIC_ENV_FILE.write_text(
            "# ==============================================================================\n"
            "# إعدادات الموديل العربي - Arabic Model Configuration\n"
            "# ==============================================================================\n"
            "\n"
            "OLLAMA_BASE_URL=http://localhost:11434\n"
            "OLLAMA_MODEL=qwen2.5:7b\n"
            "OLLAMA_LLM_MODEL=qwen2.5:7b\n"
            "OLLAMA_EMBED_MODEL=nomic-embed-text\n"
            "OLLAMA_NUM_GPU=0\n"
            "OLLAMA_TIMEOUT_SECONDS=180\n"
            "OLLAMA_NUM_CTX=4096\n"
            "OLLAMA_NUM_PREDICT=512\n"
            "OLLAMA_KEEP_ALIVE=60m\n"
            "OLLAMA_MAX_RETRIES=2\n"
            "\n"
            "LLM_BACKEND=ollama\n"
            "\n"
            "NEO4J_URI=bolt://localhost:7687\n"
            "NEO4J_USER=neo4j\n"
            "NEO4J_PASSWORD=password123\n"
            "NEO4J_TIMEOUT_SECONDS=5\n"
            "USE_GRAPH=true\n"
            "\n"
            "PDF_DIR=pdfs\n"
            "METADATA_FILE=pdfs/metadata.csv\n"
            "DATA_DIR=data\n"
            "CHROMA_PERSIST_DIR=data/chroma_db\n"
            "SQLITE_DB_PATH=data/graph_rag.db\n"
            "UPLOAD_TEMP_DIR=data/temp_uploads\n"
            "\n"
            "CHUNK_SIZE=1000\n"
            "CHUNK_OVERLAP=150\n"
            "MAX_UPLOAD_SIZE_MB=200\n"
            "\n"
            "MAX_CONTEXT_CHUNKS=5\n"
            "MAX_CONTEXT_CHARS=5000\n"
            "\n"
            "JWT_SECRET_KEY=change-this-to-a-secure-random-32-byte-hex-key\n"
            "JWT_ALGORITHM=HS256\n"
            "JWT_EXPIRY_HOURS=24\n"
            "\n"
            "RAG_EVAL_ENABLED=true\n"
            "\n"
            "APP_HOST=localhost\n"
            "APP_PORT=8501\n"
            "LOG_LEVEL=INFO\n"
            "APP_LOG_FILE=data/app.log\n"
            "\n"
            "ARABIC_MODEL_ENABLED=true\n"
            "ARABIC_PROMPT_LANGUAGE=arabic\n"
            "ARABIC_RESPONSE_LANGUAGE=arabic\n"
        )
        print("✅ تم إنشاء ملف الإعدادات العربية وتفعيل الوضع العربي (Arabic mode enabled)")

    print("\n📝 ملاحظة: أعد تشغيل التطبيق لتطبيق التغييرات")
    print("   Note: Restart the application to apply changes")


def enable_english() -> None:
    """Enable English mode by disabling .env.arabic file."""
    if ARABIC_ENV_FILE.exists():
        ARABIC_ENV_FILE.rename(ARABIC_ENV_DISABLED)
        print("✅ تم تفعيل الوضع الإنجليزي (English mode enabled)")
    elif ARABIC_ENV_DISABLED.exists():
        print("ℹ️  الوضع الإنجليزي مفعل بالفعل (English mode already enabled)")
    else:
        print("ℹ️  الوضع الإنجليزي مفعل بالفعل (English mode already enabled)")

    print("\n📝 ملاحظة: أعد تشغيل التطبيق لتطبيق التغييرات")
    print("   Note: Restart the application to apply changes")


def show_status() -> None:
    """Show current language mode status."""
    if ARABIC_ENV_FILE.exists():
        print("🌍 الوضع الحالي: عربي (Current mode: Arabic)")
        print("   ملف .env.arabic مفعل (.env.arabic is enabled)")
    elif ARABIC_ENV_DISABLED.exists():
        print("🇬🇧 الوضع الحالي: إنجليزي (Current mode: English)")
        print("   ملف .env.arabic معطل (.env.arabic is disabled)")
    else:
        print("🇬🇧 الوضع الحالي: إنجليزي (Current mode: English)")
        print("   ملف .env.arabic غير موجود (.env.arabic does not exist)")


def main() -> None:
    """Main entry point for the language switcher."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/switch_language.py arabic    # Enable Arabic mode")
        print("  python scripts/switch_language.py english   # Enable English mode")
        print("  python scripts/switch_language.py status    # Show current mode")
        print("\nالاستخدام:")
        print("  python scripts/switch_language.py arabic    # تفعيل الوضع العربي")
        print("  python scripts/switch_language.py english   # تفعيل الوضع الإنجليزي")
        print("  python scripts/switch_language.py status    # عرض الوضع الحالي")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode in ["arabic", "ar", "عربي"]:
        enable_arabic()
    elif mode in ["english", "en", "إنجليزي"]:
        enable_english()
    elif mode in ["status", "حالة"]:
        show_status()
    else:
        print(f"❌ خيار غير معروف: {mode}")
        print(f"   Unknown option: {mode}")
        print("\nالخيارات المتاحة: arabic, english, status")
        print("Available options: arabic, english, status")
        sys.exit(1)


if __name__ == "__main__":
    main()
