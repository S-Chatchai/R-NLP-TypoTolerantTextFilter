import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from pythainlp.tokenize import word_tokenize
import io

# ตั้งค่าหน้าจอ Streamlit
st.set_page_config(page_title="Fuzzy Match Filter Tool", layout="wide")
st.title("🎯 ระบบตรวจจับคำและกรองข้อมูลแบบ Dynamic N-gram")
st.write("เครื่องมือคัดกรองข้อความภาษาไทย รองรับคำพิมพ์ผิด (Typo) และเงื่อนไขแบบจับคู่ครบทุกคำในเซต")

# --- ฟังก์ชันหลักสำหรับการประมวลผลข้อความ ---
@st.cache_data
def segment_thai(text):
    if not text or pd.isna(text):
        return []
    return word_tokenize(str(text), engine='newmm')

def create_ngrams(words, n):
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ''.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def find_typo_with_ngrams_dynamic(text, keyword, fuzz_thresh=80, min_length_ratio=0.9):
    text_words = segment_thai(text)
    keyword_words = segment_thai(keyword)
    n = len(keyword_words)
    keyword_joined = ''.join(keyword_words)

    max_score = 0
    best_match = None

    for size in range(max(1, n-1), n+2):
        ngrams = create_ngrams(text_words, size)
        for ngram in ngrams:
            score = fuzz.ratio(ngram, keyword_joined)
            if len(ngram)/len(keyword_joined) >= min_length_ratio:
                if score > max_score:
                    max_score = score
                    best_match = ngram

    if max_score < fuzz_thresh:
        text_joined = ''.join(text_words)
        score = fuzz.ratio(text_joined, keyword_joined)
        if score > max_score and len(text_joined)/len(keyword_joined) >= min_length_ratio:
            max_score = score
            best_match = text_joined

    is_match = max_score >= fuzz_thresh
    return is_match, max_score, best_match

def find_typo_multiple_keyword_sets_dynamic(text, keyword_sets, fuzz_thresh=80, min_length_ratio=0.9):
    for kw_set in keyword_sets:
        matches = []
        for kw in kw_set:
            is_match, score, matched_text = find_typo_with_ngrams_dynamic(
                text, kw, fuzz_thresh=fuzz_thresh, min_length_ratio=min_length_ratio
            )
            matches.append((kw, is_match, score, matched_text))

        if all(m[1] for m in matches):
            best_score = sum(m[2] for m in matches) / len(matches)
            matched_keywords = [m[0] for m in matches]
            matched_texts = [m[3] for m in matches]
            return True, matched_keywords, best_score, matched_texts

    return False, None, 0, None

def remove_blacklist_exact(text, blacklist):
    for bl in blacklist:
        text = text.replace(bl, "")
    return text

# --- ส่วนควบคุมบนหน้าจอ UI ---
st.sidebar.header("⚙️ ตั้งค่าการกรองข้อมูล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])

kw_input = st.sidebar.text_area("ชุดคีย์เวิร์ด (1 บรรทัด = 1 เซต, ใช้คอมมาแยกคำ)", "คุณสู้เราช่วย\nไถ่ถอน, โฉนด")
blacklist_input = st.sidebar.text_input("Blacklist (แยกด้วยคอมมา)", "การ, ความ")

fuzzy_threshold = st.sidebar.slider("Fuzzy Threshold (คะแนนความคล้าย)", 50, 100, 80)
min_length_ratio = st.sidebar.slider("Min Length Ratio (สัดส่วนความยาวคำ)", 0.0, 1.0, 0.7)

# --- ส่วนการทำงานหลักเมื่อมีไฟล์อัปโหลด ---
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success(f"📂 โหลดไฟล์สำเร็จ! จำนวนทั้งหมด {len(df)} แถว")
    
    if 'รายละเอียด' not in df.columns:
        st.error("❌ ไม่พบคอลัมน์ 'รายละเอียด' ในไฟล์ที่อัปโหลด กรุณาตรวจสอบชื่อคอลัมน์")
    else:
        st.subheader("👀 ตัวอย่างข้อมูลดิบขาเข้า")
        st.dataframe(df.head(3))
        
        # จัดรูปแบบข้อมูลจาก Input หน้าเว็บ
        keyword_sets = [[k.strip() for k in line.split(",")] for line in kw_input.split("\n") if line.strip()]
        blacklist = [b.strip() for b in blacklist_input.split(",") if b.strip()]

        if st.button("▶️ เริ่มประมวลผลคัดกรองข้อมูล"):
            with st.spinner("กำลังประมวลผลข้อมูลภาษาไทย..."):
                # 1. คลีนข้อมูลและตัดคำ
                df['รายละเอียด_cleaned'] = df['รายละเอียด'].apply(lambda x: remove_blacklist_exact(str(x), blacklist))
                df['รายละเอียด_ตัดคำ'] = df['รายละเอียด_cleaned'].apply(lambda x: ' | '.join(segment_thai(str(x))))
                
                # 2. ค้นหาคำผิดด้วย Dynamic N-gram
                relevance_results = df['รายละเอียด_cleaned'].apply(
                    lambda x: find_typo_multiple_keyword_sets_dynamic(
                        str(x), keyword_sets, fuzz_thresh=fuzzy_threshold, min_length_ratio=min_length_ratio
                    )
                )
                
                #แตกผลลัพธ์ลงคอลัมน์ใหม่
                df['Relevant'] = [r[0] for r in relevance_results]
                df['Matched_keyword'] = [', '.join(r[1]) if r[1] else None for r in relevance_results]
                df['Fuzzy_score'] = [r[2] for r in relevance_results]
                df['Matched_word_in_text'] = [', '.join(r[3]) if r[3] else None for r in relevance_results]
                
                # กรองเอาเฉพาะแถวที่เกี่ยวข้อง
                df_relevant = df[df['Relevant']]
                
            st.balloons()
            st.subheader("📊 สรุปผลลัพธ์การคัดกรอง")
            col1, col2 = st.columns(2)
            col1.metric("พบข้อมูลที่เกี่ยวข้อง", f"{len(df_relevant)} แถว")
            if len(df) > 0:
                col2.metric("อัตราการจับได้ (Catch Rate)", f"{len(df_relevant)/len(df)*100:.2f}%")
            
            # ตารางพรีวิวผลลัพธ์
            st.dataframe(df_relevant[['รายละเอียด', 'Matched_keyword', 'Matched_word_in_text', 'Fuzzy_score']].head(10))
            
            # แปลงข้อมูลเป็นหน่วยความจำสำหรับดาวน์โหลด (ไม่ต้องเขียนลงดิสก์เครื่อง)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_relevant.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 ดาวน์โหลดไฟล์ Excel ผลลัพธ์",
                data=buffer.getvalue(),
                file_name="filtered_relevant_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("💡 กรุณาอัปโหลดไฟล์ Excel ที่แทบด้านซ้ายเพื่อเปิดระบบทำงาน")