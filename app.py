import os
import json
import time
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import database as db
from grader import grade_exam
from image_processor import (
    find_and_read_subject_code,
    process_single_scan_from_memory,
    read_choice_answers_final,
    read_choice_answers_50q_no_cross,
    read_exam_set_fixed,
    read_numeric_answers_advanced,
    read_student_id,
)

# streamlit run app.py

# ==========================================
# ส่วนที่ 0: ตั้งค่าแฟ้มข้อมูลและฐานข้อมูล
# ==========================================
if not os.path.exists("saved_scans"):
    os.makedirs("saved_scans")

db.init_db()

# ==========================================
# UI (Streamlit)
# ==========================================
st.set_page_config(page_title="OMR Scanner", page_icon="📄", layout="wide")
st.sidebar.title("OMR Scanner")
menu = st.sidebar.radio("เมนูหลัก", ["ตรวจข้อสอบ", "จัดการเฉลย", "ข้อมูลผู้เข้าสอบ", "ประวัติการสอบ", "คลังเอกสาร"])


# ==========================================
# หน้า 1: ตรวจข้อสอบ
# ==========================================
if menu == "ตรวจข้อสอบ":
    st.header("ระบบตรวจข้อสอบอัตโนมัติ (OMR)")
    with st.container(border=True):
        st.subheader("อัปโหลดกระดาษคำตอบเพื่อประมวลผล")
        uploaded_files = st.file_uploader("ลากไฟล์กระดาษคำตอบลงที่นี่ (.jpg, .png)", accept_multiple_files=True)

        if st.button("🚀 เริ่มตรวจข้อสอบ", type="primary"):
            if uploaded_files:
                my_bar = st.progress(0, text="กำลังประมวลผล...")

                for idx, file in enumerate(uploaded_files):
                    file_path = os.path.join("saved_scans", file.name)
                    with open(file_path, "wb") as f: f.write(file.read())
                    file.seek(0)

                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    raw_img = cv2.imdecode(file_bytes, 1)
                    final_output = process_single_scan_from_memory(raw_img)

                    if final_output is not None:
                        subject_code = find_and_read_subject_code(final_output)
                        student_id = read_student_id(final_output)
                        exam_set = f"ชุดที่ {read_exam_set_fixed(final_output)}"

                        student_data = db.get_student(student_id)
                        student_name = f"{student_data[0]} {student_data[1]}" if student_data else "ไม่พบรายชื่อในระบบ"

                        db_result = db.get_answer_key(subject_code, exam_set)
                        score, wrong_questions, empty_questions, double_questions, exam_title = grade_exam(final_output, db_result)

                        if db_result is None:
                            st.error(f"⚠️ ไฟล์ {file.name}: ไม่พบเฉลยของรหัสวิชา '{subject_code}' {exam_set}")

                        # แสดงผลสรุปรายไฟล์
                        with st.expander(f"📄 {file.name} — {student_name} | คะแนน: {score:g}"):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                wrong_only = [q for q in wrong_questions if q not in empty_questions and q not in double_questions]
                                st.markdown(f"**❌ ข้อที่ตอบผิด ({len(wrong_only)} ข้อ):**")
                                st.write(", ".join(wrong_only) if wrong_only else "-")
                            with c2:
                                st.markdown(f"**⬜ ข้อที่ไม่ตอบ ({len(empty_questions)} ข้อ):**")
                                if empty_questions:
                                    st.warning(", ".join(empty_questions))
                                else:
                                    st.write("-")
                            with c3:
                                st.markdown(f"**⚠️ ข้อที่ตอบ 2 ตัวเลือก ({len(double_questions)} ข้อ):**")
                                if double_questions:
                                    st.warning(", ".join(double_questions))
                                else:
                                    st.write("-")

                        wrong_str = ", ".join(wrong_questions) if wrong_questions else "-"
                        empty_str = ", ".join(empty_questions) if empty_questions else "-"
                        double_str = ", ".join(double_questions) if double_questions else "-"
                        db.save_exam_result(file.name, student_id, student_name, subject_code, exam_title, exam_set,
                                            score, wrong_str, empty_str, double_str)

                    my_bar.progress((idx + 1) / len(uploaded_files), text=f"ตรวจแล้ว {idx+1}/{len(uploaded_files)}")
                st.success("✅ ตรวจข้อสอบและบันทึกประวัติเสร็จสิ้น! สามารถดูผลลัพธ์ได้ที่เมนู 'ประวัติการสอบ'")
            else:
                st.warning("กรุณาอัปโหลดไฟล์ก่อนครับ")


# ==========================================
# หน้า 2: จัดการเฉลย
# ==========================================
elif menu == "จัดการเฉลย":
    st.header("🎯 จัดการเฉลยข้อสอบ")

    with st.container(border=True):
        st.subheader("📝 สร้างหรืออัปเดตเฉลยใหม่")
        col1, col2, col3 = st.columns(3)
        with col1: subj_code = st.text_input("รหัสวิชา (2 หลัก) *", placeholder="เช่น 01", max_chars=2)
        with col2: subj_name = st.text_input("ชื่อวิชา/การสอบ *", placeholder="เช่น คณิตศาสตร์ กลางภาค")
        with col3: exam_set = st.selectbox("ชุดข้อสอบ *", ["ชุดที่ 1", "ชุดที่ 2"])

        exam_format = st.radio("รูปแบบข้อสอบ", ["ปรนัย 25 ข้อ + อัตนัย 5 ข้อ", "ปรนัย 50 ข้อ"], horizontal=True)

        if ('draft_format' not in st.session_state or st.session_state.draft_format != exam_format):
            st.session_state.draft_format = exam_format
            if exam_format == "ปรนัย 25 ข้อ + อัตนัย 5 ข้อ":
                st.session_state.draft_answers = ["1"] * 30
                st.session_state.draft_scores = [1.0] * 25 + [5.0] * 5
            else:
                st.session_state.draft_answers = ["1"] * 50
                st.session_state.draft_scores = [1.0] * 50

        master_sheet = st.file_uploader("📷 สแกนจากกระดาษเฉลย (Master Sheet)", type=['jpg', 'png'])
        if st.button("🔍 สแกนเฉลยอัตโนมัติ"):
            if master_sheet:
                file_bytes = np.asarray(bytearray(master_sheet.read()), dtype=np.uint8)
                final_output = process_single_scan_from_memory(cv2.imdecode(file_bytes, 1))
                if final_output is not None:
                    if exam_format == "ปรนัย 25 ข้อ + อัตนัย 5 ข้อ":
                        ans_c = read_choice_answers_final(final_output)
                        ans_n = read_numeric_answers_advanced(final_output)
                        new_ans = [str(ans_c.get(i, "?")) for i in range(1, 26)] + [str(ans_n.get(i, "?")) for i in range(26, 31)]
                    else:
                        ans_c = read_choice_answers_50q_no_cross(final_output)
                        new_ans = [str(ans_c.get(i, "?")) for i in range(1, 51)]
                    st.session_state.draft_answers = new_ans
                    st.success("✅ สแกนสำเร็จ!")
                else:
                    st.error("❌ ไม่พบขอบกระดาษ")

        df_answers = pd.DataFrame({
            "ข้อที่": list(range(1, len(st.session_state.draft_answers) + 1)),
            "คำตอบ": st.session_state.draft_answers,
            "คะแนน": st.session_state.draft_scores
        })

        st.write("✏️ **แก้ไขคำตอบและคะแนนในตารางได้ทันที:**")
        edited_df = st.data_editor(df_answers, use_container_width=True, hide_index=True)

        total_score = edited_df["คะแนน"].astype(float).sum()
        st.info(f"🏆 คะแนนเต็มรวม: **{total_score:g}** คะแนน")

        if st.button("💾 บันทึกเฉลยลงระบบ", type="primary", use_container_width=True):
            if subj_code and subj_name:
                ans_list = edited_df["คำตอบ"].astype(str).tolist()
                score_list = edited_df["คะแนน"].astype(float).tolist()
                key_data = [{"answer": a, "score": s} for a, s in zip(ans_list, score_list)]
                db.save_answer_key(subj_code, exam_set, subj_name, exam_format, key_data)
                st.success(f"✅ บันทึก '{subj_name}' เรียบร้อย!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ กรุณากรอกรหัสและชื่อวิชา")

    st.divider()
    st.subheader("📚 รายการเฉลยในระบบ")
    df_db = db.get_all_answer_keys()

    if not df_db.empty:
        st.dataframe(df_db.drop(columns=['key_data']), use_container_width=True, hide_index=True)

        col_sel, col_view, col_exp = st.columns([2, 1, 1])
        with col_sel:
            target = st.selectbox("เลือกวิชาเพื่อดู/ส่งออก:", [f"{r['subject_code']} | {r['exam_set']}" for _, r in df_db.iterrows()])
            sel_code, sel_set = target.split(" | ")
            current_row = df_db[(df_db['subject_code'] == sel_code) & (df_db['exam_set'] == sel_set)].iloc[0]
            current_keys = json.loads(current_row['key_data'])
            df_export = pd.DataFrame(current_keys)
            df_export.index = df_export.index + 1
            df_export.index.name = "ข้อที่"

        with col_view:
            st.write(""); st.write("")
            if st.button("👁️ ดูรายละเอียด", use_container_width=True):
                st.table(df_export.rename(columns={"answer": "คำตอบ", "score": "คะแนน"}))

        with col_exp:
            st.write(""); st.write("")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, sheet_name='AnswerKey')
            excel_data = output.getvalue()
            st.download_button(label="📥 โหลด Excel", data=excel_data, file_name=f"Key_{sel_code}_{sel_set}.xlsx", mime="application/vnd.ms-excel", use_container_width=True)

        if st.button(f"🗑️ ลบเฉลย {target}", type="secondary"):
            db.delete_answer_key(sel_code, sel_set)
            st.rerun()
    else:
        st.info("ยังไม่มีข้อมูลเฉลย")


# ==========================================
# หน้า 3: ข้อมูลผู้เข้าสอบ
# ==========================================
elif menu == "ข้อมูลผู้เข้าสอบ":
    st.header("👥 จัดการข้อมูลผู้เข้าสอบ")

    search = st.text_input("🔍 ค้นหาด้วย เลขที่นั่งสอบ, ชื่อ หรือ นามสกุล", placeholder="พิมพ์เพื่อค้นหา...")
    tab1, tab2, tab3 = st.tabs(["📋 รายชื่อทั้งหมด", "➕ เพิ่มนักเรียน", "📁 นำเข้าไฟล์"])

    with tab1:
        df_students = db.get_students()
        if search:
            mask = (df_students["seat_number"].astype(str).str.contains(search, case=False) |
                    df_students["first_name"].str.contains(search, case=False) |
                    df_students["last_name"].str.contains(search, case=False))
            df_students = df_students[mask]

        st.write("📝 **แก้ไขข้อมูล:** แก้ไขชื่อ-นามสกุลได้โดยตรงในตาราง")
        edited_df = st.data_editor(
            df_students,
            hide_index=True,
            column_config={
                "seat_number": st.column_config.TextColumn("เลขที่นั่งสอบ (7 หลัก)", disabled=True),
                "first_name": st.column_config.TextColumn("ชื่อ"),
                "last_name": st.column_config.TextColumn("นามสกุล")
            },
            use_container_width=True,
            key="edit_table"
        )

        if not edited_df.equals(df_students):
            if st.button("💾 บันทึกการเปลี่ยนแปลงชื่อ", use_container_width=True):
                for _, row in edited_df.iterrows():
                    db.update_student(row['seat_number'], row['first_name'], row['last_name'])
                st.success("อัปเดตเรียบร้อย!")
                time.sleep(0.5)
                st.rerun()

        st.divider()
        st.subheader("🗑️ ลบรายชื่อนักเรียน")
        delete_list = st.multiselect(
            "เลือกเลขที่นั่งสอบที่ต้องการลบ (เลือกได้หลายคน):",
            options=df_students["seat_number"].tolist(),
            format_func=lambda x: f"รหัส {x} - {df_students[df_students['seat_number']==x]['first_name'].values[0]}"
        )

        if delete_list:
            if st.button(f"ยืนยันการลบ {len(delete_list)} รายการ", type="primary", use_container_width=True):
                db.delete_students(delete_list)
                st.success("ลบข้อมูลสำเร็จ!")
                time.sleep(0.5)
                st.rerun()

    with tab2:
        with st.form("add_form", clear_on_submit=True):
            s_id = st.text_input("เลขที่นั่งสอบ (7 หลัก)", max_chars=7)
            f_n = st.text_input("ชื่อ")
            l_n = st.text_input("นามสกุล")
            if st.form_submit_button("บันทึก"):
                if len(s_id) == 7:
                    try:
                        db.add_student(s_id, f_n, l_n)
                        st.rerun()
                    except:
                        st.error("รหัสซ้ำ!")
                else:
                    st.warning("กรุณากรอกรหัส 7 หลัก")

    with tab3:
        student_file = st.file_uploader("นำเข้าไฟล์", type=['csv', 'xlsx'])
        if st.button("อัปโหลด"):
            if student_file:
                df_imp = pd.read_csv(student_file) if student_file.name.endswith('.csv') else pd.read_excel(student_file)
                db.import_students(df_imp)
                st.success("สำเร็จ!")
                st.rerun()


# ==========================================
# หน้า 4: ประวัติการสอบ
# ==========================================
elif menu == "ประวัติการสอบ":
    st.header("ประวัติและสถิติคะแนนสอบ")

    df_history = db.get_exam_history()

    if not df_history.empty:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_hist = st.text_input("🔍 ค้นหาด้วย เลขที่, ชื่อ, หรือ ไฟล์")
        with col2:
            subject_list = ["ทั้งหมด"] + df_history["วิชาที่สอบ"].unique().tolist()
            filter_subj = st.selectbox("กรองตามวิชา", subject_list)
        with col3:
            set_list = ["ทั้งหมด"] + df_history["ชุด"].unique().tolist()
            filter_set = st.selectbox("ชุดข้อสอบ", set_list)

        if filter_subj != "ทั้งหมด":
            df_history = df_history[df_history["วิชาที่สอบ"] == filter_subj]
        if filter_set != "ทั้งหมด":
            df_history = df_history[df_history["ชุด"] == filter_set]
        if search_hist:
            mask = (
                df_history["เลขที่นั่งสอบ"].astype(str).str.contains(search_hist, case=False, na=False) |
                df_history["ชื่อนักเรียน"].str.contains(search_hist, case=False, na=False) |
                df_history["ชื่อไฟล์"].str.contains(search_hist, case=False, na=False)
            )
            df_history = df_history[mask]

        st.write(f"📊 พบข้อมูลประวัติการสอบจำนวน **{len(df_history)}** รายการ")
        dl_col1, dl_col2 = st.columns([4, 1])
        with dl_col2:
            csv = df_history.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 ส่งออกเป็น CSV", data=csv, file_name="exam_history.csv", mime="text/csv", use_container_width=True)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        st.divider()
        if st.button("🗑️ ล้างประวัติการสอบทั้งหมด (Clear All)", type="secondary"):
            db.clear_exam_history()
            st.success("ล้างประวัติสำเร็จ!")
            time.sleep(1)
            st.rerun()
    else:
        st.info("ยังไม่มีข้อมูลประวัติการสอบ ไปที่เมนู 'ตรวจข้อสอบ' เพื่อเริ่มใช้งานครับ")


# ==========================================
# หน้า 5: คลังเอกสาร
# ==========================================
elif menu == "คลังเอกสาร":
    st.header("คลังเอกสารและกระดาษคำตอบ")

    if not os.path.exists("templates"):
        os.makedirs("templates")

    tab1, tab2 = st.tabs(["📄 เอกสารต้นแบบ (สำหรับปริ้นท์/ใช้งาน)", "📁 กระดาษที่ตรวจแล้ว (ไฟล์สแกน)"])

    with tab1:
        st.subheader("จัดการเอกสารต้นแบบ (Templates & Forms)")
        st.markdown("อัปโหลดไฟล์กระดาษคำตอบเปล่า หรือไฟล์ Excel ต้นแบบ (PDF, JPG, PNG, XLS, XLSX, CSV) ไว้ที่นี่ เพื่อให้ง่ายต่อการดาวน์โหลดไปใช้งาน")

        with st.form("upload_template_form", clear_on_submit=True):
            new_template = st.file_uploader("อัปโหลดไฟล์เอกสารต้นแบบ", type=['pdf', 'jpg', 'png', 'xls', 'xlsx', 'csv'])
            submitted = st.form_submit_button("💾 บันทึกไฟล์ต้นแบบเข้าสู่ระบบ")
            if submitted and new_template:
                with open(os.path.join("templates", new_template.name), "wb") as f:
                    f.write(new_template.read())
                st.success(f"✅ อัปโหลดไฟล์ {new_template.name} สำเร็จ!")
                time.sleep(1)
                st.rerun()

        st.divider()
        st.subheader("📥 ไฟล์ต้นแบบพร้อมดาวน์โหลด")
        template_files = os.listdir("templates")
        if template_files:
            for f in template_files:
                col_file, col_btn, col_del = st.columns([3, 1, 1])
                with col_file:
                    st.write(f"📄 **{f}**")
                with col_btn:
                    with open(os.path.join("templates", f), "rb") as file_data:
                        st.download_button("📥 ดาวน์โหลด", data=file_data, file_name=f, key=f"dl_{f}", use_container_width=True)
                with col_del:
                    if st.button("🗑️ ลบ", key=f"del_{f}", use_container_width=True):
                        os.remove(os.path.join("templates", f))
                        st.success("ลบสำเร็จ")
                        time.sleep(1)
                        st.rerun()
        else:
            st.info("ยังไม่มีไฟล์เอกสารต้นแบบ กรุณาอัปโหลดไฟล์ที่ด้านบนครับ")

    with tab2:
        st.subheader("กระดาษคำตอบที่สแกนแล้ว (Scanned Files)")
        saved_files = os.listdir("saved_scans") if os.path.exists("saved_scans") else []

        if saved_files:
            file_data = []
            for f in saved_files:
                file_path = os.path.join("saved_scans", f)
                size_kb = os.path.getsize(file_path) / 1024
                date_m = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                file_data.append({"ชื่อไฟล์": f, "ขนาด (KB)": f"{size_kb:.1f}", "เวลาที่บันทึก": date_m})
            st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)

            st.write("---")
            col_dl, col_del_scan = st.columns(2)

            with col_dl:
                st.subheader("ดาวน์โหลดไฟล์ต้นฉบับ")
                dl_file = st.selectbox("เลือกไฟล์ที่ต้องการโหลด", saved_files, key="dl_scan_select")
                with open(os.path.join("saved_scans", dl_file), "rb") as f:
                    st.download_button("📥 ดาวน์โหลดไฟล์นี้", data=f, file_name=dl_file)

            with col_del_scan:
                st.subheader("ลบไฟล์สแกน")
                del_scan_file = st.selectbox("เลือกไฟล์ที่ต้องการลบ", saved_files, key="del_scan_select")
                if st.button("🗑️ ลบไฟล์นี้", type="secondary"):
                    os.remove(os.path.join("saved_scans", del_scan_file))
                    db.delete_history_by_filename(del_scan_file)
                    st.success("ลบไฟล์และประวัติสำเร็จ!")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("ยังไม่มีไฟล์กระดาษคำตอบในระบบ")
