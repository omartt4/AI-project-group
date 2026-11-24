import json
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "chunks" / "chunks.jsonl"
QUESTIONS_PATH = ROOT / "eval" / "questions_ar.jsonl"
PAGE_COUNTS = {"140": 204, "1144": 246}

QUESTIONS = [
    {
        "question_id": "ar_140_01",
        "book_id": "140",
        "source_page_index": 10,
        "question": "ما معنى كلمة السلف في الاصطلاح؟",
        "answer_quote": "هم أصحاب النبي ﷺ ومن تبعهم وسار على طريقتهم",
    },
    {
        "question_id": "ar_140_02",
        "book_id": "140",
        "source_page_index": 14,
        "question": "كيف يصف أهل السنة والجماعة الله تعالى في باب الأسماء والصفات؟",
        "answer_quote": "من غير تعطيل ولا تأويل ومن غير تمثيل ولا تكييف",
    },
    {
        "question_id": "ar_140_03",
        "book_id": "140",
        "source_page_index": 23,
        "question": "ما موقف أهل السنة من الخروج على الحاكم المسلم؟",
        "answer_quote": "يحرم الخروج عليه سواء كان تقيًا أو عاصيًا",
    },
    {
        "question_id": "ar_140_04",
        "book_id": "140",
        "source_page_index": 59,
        "question": "ماذا تنفي عبارة «لا إله» في كلمة التوحيد؟",
        "answer_quote": "وهو نفي الإلهية عن كل ما سوى الله تعالى",
    },
    {
        "question_id": "ar_140_05",
        "book_id": "140",
        "source_page_index": 68,
        "question": "كيف تتأثر عبادة العبد إذا ضعفت محبة الله في قلبه؟",
        "answer_quote": "فتصبح عباداته أقرب إلى العادة منها إلى العبادة",
    },
    {
        "question_id": "ar_140_06",
        "book_id": "140",
        "source_page_index": 122,
        "question": "لماذا يكون المنافقون أشد عذابًا من سائر الكفار؟",
        "answer_quote": "لأنهم زادوا على الكفر: الكذب والمرواغة والخداع",
    },
    {
        "question_id": "ar_140_07",
        "book_id": "140",
        "source_page_index": 131,
        "question": "ما أول وأعظم سبب أوقع بني آدم في الشرك الأكبر؟",
        "answer_quote": "وثبت أن الغلو في الصالحين كان هو أول وأعظم سبب أوقع بني آدم في الشرك الأكبر",
    },
    {
        "question_id": "ar_140_08",
        "book_id": "140",
        "source_page_index": 149,
        "question": "ما حكم من أراد بعبادته وجه الله والدنيا معًا؟",
        "answer_quote": "فهذا الأقرب أنه مباح؛ لأن الوعيد إنما ورد في حق من طلب بالعبادة الدنيا وحدها",
    },
    {
        "question_id": "ar_140_09",
        "book_id": "140",
        "source_page_index": 158,
        "question": "ما تعريف الحلف في الاصطلاح الشرعي؟",
        "answer_quote": "توكيد الشيء بذكر اسم أو صفة لله تعالى مصدرًا بحرف من حروف القسم",
    },
    {
        "question_id": "ar_140_10",
        "book_id": "140",
        "source_page_index": 167,
        "question": "ما الخصلة الرابعة من خصال النفاق الأصغر؟",
        "answer_quote": "أن يعاهد غيره بعهد، وفي نيته وقت العهد أن لا يفي به",
    },
    {
        "question_id": "ar_140_11",
        "book_id": "140",
        "source_page_index": 176,
        "question": "ما الحكم الشرعي لصلاة الرغائب في شهر رجب؟",
        "answer_quote": "وكل هذا لا دليل عليه، وهو من البدع المحرمة",
    },
    {
        "question_id": "ar_1144_01",
        "book_id": "1144",
        "source_page_index": 10,
        "question": "ما العلاقة بين القضاء والقدر عند ابن عثيمين إذا اجتمعا وإذا افترقا؟",
        "answer_quote": "القضاء والقدر متباينان إن اجتمعا، ومترادفان إن افترقا",
    },
    {
        "question_id": "ar_1144_02",
        "book_id": "1144",
        "source_page_index": 25,
        "question": "ما قول الأشعرية في فعل العبد؟",
        "answer_quote": "ينفون عن العبد الفعل، ويثبتون له قدرةً غير مؤثِّرةٍ، وكسبًا",
    },
    {
        "question_id": "ar_1144_03",
        "book_id": "1144",
        "source_page_index": 55,
        "question": "كيف رد أهل السنة على استدلال الجبرية بآية رمي النبي بكف من تراب؟",
        "answer_quote": "فبطل زعم الجبريَّة، ودلَّت الآية على مذهب أهل السنَّة",
    },
    {
        "question_id": "ar_1144_04",
        "book_id": "1144",
        "source_page_index": 65,
        "question": "ما دعوى القدرية في معنى الهداية من الله تعالى كما نقل البغدادي؟",
        "answer_quote": "أن الهداية من الله تعالى على معنى الإرشاد والدعاء إلى الحق",
    },
    {
        "question_id": "ar_1144_05",
        "book_id": "1144",
        "source_page_index": 75,
        "question": "في أي شيء أصابت القدرية وفي أي شيء أصابت الجبرية؟",
        "answer_quote": "والقدريَّة مصيبون في أن الله لا يحبُّ الكفر والمعاصي، والجبريَّة مصيبون في أن الله قدَّرها وشاءها",
    },
    {
        "question_id": "ar_1144_06",
        "book_id": "1144",
        "source_page_index": 85,
        "question": "ما المقصود بالأصلح في مسألة الصلاح والأصلح؟",
        "answer_quote": "والأصلح: الأنفع والأحسن لهم",
    },
    {
        "question_id": "ar_1144_07",
        "book_id": "1144",
        "source_page_index": 115,
        "question": "ما حقيقة الظلم الذي ينزه الله تعالى عنه عند أهل السنة؟",
        "answer_quote": "وهو أن حقيقة الظلم المنفيِّ عن الله تعالى: وضع الأشياء في غير مواضعها، مع اتِّصافه بكمال ضدِّه",
    },
    {
        "question_id": "ar_1144_08",
        "book_id": "1144",
        "source_page_index": 155,
        "question": "ماذا قال الإمام أحمد عن الجدال في مسائل القدر؟",
        "answer_quote": "فإن الكلام في القدر والرؤية والقرآن وغيرها من السنن مكروهٌ ومنهيٌّ عنه",
    },
    {
        "question_id": "ar_1144_09",
        "book_id": "1144",
        "source_page_index": 205,
        "question": "ما الحكم على من كذب بالقدر كما نُقل عن الحسن؟",
        "answer_quote": "من كذَّب بالقدر فقد كذب بالقرآن",
    },
    {
        "question_id": "ar_1144_10",
        "book_id": "1144",
        "source_page_index": 215,
        "question": "متى ولد قتادة بن دعامة السدوسي؟",
        "answer_quote": "كانت ولادته سنة ستين للهجرة",
    },
    {
        "question_id": "ar_1144_11",
        "book_id": "1144",
        "source_page_index": 235,
        "question": "ما الصورة التي لا تتعلق بها الإرادة الكونية ولا الشرعية، وما مثالها؟",
        "answer_quote": "كفر المؤمن والمعاصي التي عصمه الله منها فلم تقع منه",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    if QUESTIONS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {QUESTIONS_PATH}")
    chunks = load_jsonl(CHUNKS_PATH)
    chunks_by_book = {
        book_id: [
            chunk for chunk in chunks if str(chunk["book_id"]) == book_id
        ]
        for book_id in PAGE_COUNTS
    }
    cleaned_by_book = {
        book_id: (ROOT / "cleaned" / f"book_{book_id}.txt").read_text(
            encoding="utf-8"
        )
        for book_id in PAGE_COUNTS
    }

    completed = []
    seen_question_ids = set()
    for supplied in QUESTIONS:
        book_id = supplied["book_id"]
        if supplied["question_id"] in seen_question_ids:
            raise ValueError(f"Duplicate question ID: {supplied['question_id']}")
        seen_question_ids.add(supplied["question_id"])
        if not 0 <= supplied["source_page_index"] < PAGE_COUNTS[book_id]:
            raise ValueError(
                f"Page index out of range for {supplied['question_id']}."
            )

        quote = unicodedata.normalize("NFC", supplied["answer_quote"])
        text = cleaned_by_book[book_id]
        occurrences = text.count(quote)
        if occurrences != 1:
            raise ValueError(
                f"{supplied['question_id']}: answer_quote occurs {occurrences} "
                "times in the cleaned book; exact gold span is not unique."
            )
        gold_start = text.find(quote)
        gold_end = gold_start + len(quote)
        center = (gold_start + gold_end) // 2
        matching_chunks = [
            chunk
            for chunk in chunks_by_book[book_id]
            if chunk["char_start"] <= center <= chunk["char_end"]
        ]
        if not matching_chunks:
            raise ValueError(
                f"{supplied['question_id']}: no chunk contains the gold span center."
            )
        full_span_chunks = [
            chunk
            for chunk in matching_chunks
            if chunk["char_start"] <= gold_start
            and gold_end <= chunk["char_end"]
        ]
        selected_chunk = (full_span_chunks or matching_chunks)[0]
        completed.append(
            {
                **supplied,
                "answer_quote": quote,
                "chunk_id": selected_chunk["chunk_id"],
                "gold_start": gold_start,
                "gold_end": gold_end,
            }
        )

    if len(completed) != 22:
        raise ValueError(f"Expected 22 supplied questions, got {len(completed)}.")
    with QUESTIONS_PATH.open("x", encoding="utf-8", newline="\n") as output:
        for question in completed:
            output.write(json.dumps(question, ensure_ascii=False) + "\n")
    print(f"questions_saved={len(completed)}")
    print(f"book_140={sum(q['book_id'] == '140' for q in completed)}")
    print(f"book_1144={sum(q['book_id'] == '1144' for q in completed)}")
    print(f"questions_file={QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
