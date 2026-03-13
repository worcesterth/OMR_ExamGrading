import json
from image_processor import (
    read_choice_answers_final,
    read_numeric_answers_advanced,
    read_choice_answers_50q_no_cross,
)


def grade_exam(final_output, db_result):
    """
    Grade a scanned answer sheet against an answer key from the database.

    Args:
        final_output: Processed image (numpy array)
        db_result: Tuple of (exam_format, key_data_json, exam_title) from DB, or None

    Returns:
        (score: float, wrong_questions: list[str], empty_questions: list[str],
         double_questions: list[str], exam_title: str)
    """
    score = 0.0
    wrong_questions = []
    empty_questions = []
    double_questions = []
    exam_title = "ไม่พบวิชาในระบบ"

    if db_result is None:
        return score, wrong_questions, empty_questions, double_questions, exam_title

    exam_format, key_data_json, exam_title = db_result
    master_data = json.loads(key_data_json)

    if len(master_data) > 0 and isinstance(master_data[0], dict):
        master_answers = [str(item['answer']) for item in master_data]
        master_scores = [float(item['score']) for item in master_data]
    else:
        # Fallback for old data format (plain list of strings)
        master_answers = [str(a) for a in master_data]
        master_scores = [
            (2.5 if i < 25 else 5.0) if exam_format == "ปรนัย 25 ข้อ + อัตนัย 5 ข้อ" else 2.0
            for i in range(len(master_answers))
        ]

    student_answers = []
    if exam_format == "ปรนัย 25 ข้อ + อัตนัย 5 ข้อ":
        ans_choices = read_choice_answers_final(final_output)
        ans_numeric = read_numeric_answers_advanced(final_output)
        for i in range(1, 26): student_answers.append(str(ans_choices.get(i, "?")))
        for i in range(26, 31): student_answers.append(str(ans_numeric.get(i, "?")))
    else:
        ans_choices = read_choice_answers_50q_no_cross(final_output)
        for i in range(1, 51): student_answers.append(str(ans_choices.get(i, "?")))

    for i in range(min(len(master_answers), len(student_answers))):
        ans = student_answers[i]
        q_num = str(i + 1)
        if ans == "Empty":
            empty_questions.append(q_num)
            wrong_questions.append(q_num)
        elif ans == "Double":
            double_questions.append(q_num)
            wrong_questions.append(q_num)
        elif ans == master_answers[i]:
            score += master_scores[i]
        else:
            wrong_questions.append(q_num)

    return score, wrong_questions, empty_questions, double_questions, exam_title
