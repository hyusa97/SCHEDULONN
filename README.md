"""
Streamlit app: AI-Assisted Timetable Generator (CP-SAT backend)
------------------------------------------------------------
Single-file Streamlit app that:
- Accepts CSV/Excel uploads for courses, sections, faculties, rooms, fixed events
- Lets you define days & slots per day interactively
- Calls the CP-SAT solver (OR-Tools) to generate a clash-free timetable
- Shows result in-app and allows Excel/CSV download


Dependencies:
pip install streamlit pandas ortools openpyxl reportlab


Run:
streamlit run this_file.py


Notes:
- This file embeds the solver previously created. Keep it as a single-file prototype for quick deployment.
- For production, split solver code into modules and add DB-backed storage, authentication, and better UX.
"""
