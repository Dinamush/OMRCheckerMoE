Prefill Answer Sheet Package
============================

Purpose
-------
Pre-fills the landscape OMR answer sheet template with:
  - Student Name
  - School Name
  - Exam Name
  - Candidate Number (10-digit bubble grid)

Files
-----
blank_template_reference.png  -- Blank answer sheet template (do not modify)
prefill_answer_sheet_final.py -- Prefill script (single sheet and batch modes)
sample_students.csv           -- Example batch input CSV
johnathan_sample_prefilled.png -- Reference output for visual alignment checks
sample_outputs/               -- Sample PNG outputs for reference
outputs/                      -- Output directory (populated by batch runs)

Recommended usage
-----------------
Use the web UI at http://127.0.0.1:8000/prefill after starting the server:

    uvicorn webui.app:create_app --factory --reload

The UI supports:
  - Single sheet: fill the form and download PNG or PDF
  - Batch: upload a CSV file or enter rows in the inline table, download
           as a combined PDF or a ZIP of individual PNGs

Command-line usage
------------------
Single sheet (PNG output):

    python prefill_answer_sheet_final.py ^
        --template blank_template_reference.png ^
        --student-name "Johnathan Ragnauth Brigmohan" ^
        --school-name "The New Sapodilla Primary" ^
        --exam-name "National Grade Four Reading Assessment" ^
        --candidate-number 9010690012 ^
        --output johnathan_sheet.png

Single sheet (PDF output): use --output with a .pdf extension.

Batch from CSV (combined PDF + individual PNGs):

    python prefill_answer_sheet_final.py ^
        --template blank_template_reference.png ^
        --csv sample_students.csv ^
        --combined-pdf all_sheets.pdf ^
        --output-dir outputs

CSV format
----------
Required columns: student_name, school_name, exam_name, candidate_number
Optional column:  output_file (PNG filename used when --output-dir is set)

Example:
    student_name,school_name,exam_name,candidate_number,output_file
    Johnathan Ragnauth Brigmohan,The New Sapodilla Primary,Grade 4 Reading,9010690012,johnathan.png

Notes
-----
- Candidate number must be exactly 10 digits.
- Bubble placement is tuned to blank_template_reference.png. Do not swap
  in a different template without re-calibrating the CFG values in the script.
