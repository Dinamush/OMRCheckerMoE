Prefill-Only Answer Sheet Package

Included files
--------------
- blank_template_reference.png
- prefill_answer_sheet_final.py
- prefill_tuning.json
- sample_students.csv
- johnathan_sample_prefilled.png
- sample_students_combined.pdf
- sample_outputs/

Purpose
-------
This package contains only the files needed to prefill answer sheets with:
- Student Name
- School Name
- Exam Name
- Candidate Number

Windows commands
----------------
Single student:
python prefill_answer_sheet_corrected.py --template blank_template_reference.png --student-name "Johnathan Ragnauth Brigmohan" --school-name "The New Sapodilla Primary" --exam-name "National Grade Four Reading Assessment" --candidate-number 9010690012 --output johnathan_sheet.png

Batch:
python prefill_answer_sheet_corrected.py --template blank_template_reference.png --csv sample_students.csv --combined-pdf all_sheets.pdf --output-dir outputs
